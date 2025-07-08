import logging
import pandas as pd

# Local module imports
import utils as u
import utilities.bq_utils as bqu

# set up logger at the module level
logger = logging.getLogger(__name__)



# --------------------------------------
#        Features Main Interface
# --------------------------------------

@u.timing_decorator
def calculate_transfers_features(
    profits_df: pd.DataFrame,
    transfers_sequencing_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Retrieves facts about the wallet's transfer activity based on blockchain data.
    Period boundaries are defined by the dates in profits_df through the inner join.

    Params:
    - profits_df (df): the profits_df for the period that the features will reflect,
        indexed on: [coin_id, wallet_address, date]
    - transfers_sequencing_df (df): each wallet's lifetime transfers data

    Returns:
    - transfers_features_df (df): dataframe indexed on wallet_address with
        transfers feature columns
    """
    profits_df = u.ensure_index(profits_df)

    # Calculate current total buyers of each coin and compare with each wallet's buyer number
    total_buyers = transfers_sequencing_df.groupby('coin_id', observed=True)['buyer_number'].max()
    total_buyers.name = 'coin_total_buyers'
    transfers_sequencing_df = transfers_sequencing_df.merge(total_buyers, left_on='coin_id', right_index=True)
    transfers_sequencing_df['later_buyers'] = (transfers_sequencing_df['coin_total_buyers']
                                                    - transfers_sequencing_df['buyer_number'])
    transfers_sequencing_df['later_buyers_ratio'] = u.winsorize(
            (transfers_sequencing_df['coin_total_buyers']
            / transfers_sequencing_df['buyer_number'])
        ,0.01)

    # Calculate initial_hold_time in days. If no sells, use the latest date in profits_df.
    latest_date = profits_df.index.get_level_values('date').max()
    transfers_sequencing_df['initial_hold_time'] = (
        transfers_sequencing_df['first_sell'].fillna(latest_date) - transfers_sequencing_df['first_buy']
    ).dt.days

    # Filter to only coins and wallets during the profits_df date range
    first_buy_data = pd.merge(
        profits_df,
        transfers_sequencing_df,
        left_index=True,
        right_on=['coin_id', 'wallet_address', 'first_buy'],
        how='inner'
    )

    # Aggregate to features
    transfers_features_df = first_buy_data.groupby('wallet_address', observed=True).agg(
    **{
        'first_buy/mean_rank':          ('buyer_number', 'mean'),
        'first_buy/median_rank':        ('buyer_number', 'median'),
        'first_buy/earliest_rank':      ('buyer_number', 'min'),
        'initial_hold_time/mean':       ('initial_hold_time', 'mean'),
        'initial_hold_time/median':     ('initial_hold_time', 'median'),
        'buys_coin_age/mean':           ('coin_age', 'mean'),
        'buys_coin_age/median':         ('coin_age', 'median'),
        'buys_coin_age/min':            ('coin_age', 'min'),
        'coin_total_buyers/mean':       ('coin_total_buyers', 'mean'),
        'coin_total_buyers/median':     ('coin_total_buyers', 'median'),
        'later_buyers/mean':            ('later_buyers', 'mean'),
        'later_buyers/median':          ('later_buyers', 'median'),
        'earlybird_ratio/mean':         ('later_buyers_ratio', 'mean'),
        'earlybird_ratio/median':       ('later_buyers_ratio', 'median'),
        'wallet_age':                   ('wallet_age', 'first')  # all wallet-grouped values will be identical
    })

    # Data quality validation - all features should be positive
    feature_cols = transfers_features_df.columns
    for col in feature_cols:
        negative_count = (transfers_features_df[col] < 0).sum()
        if negative_count > 0:
            logger.error(f"Found {negative_count} negative values in feature {col}")
            raise ValueError(f"Feature validation failed: {col} contains negative values")

    return transfers_features_df



# -----------------------------------
#       Data Retrieval Function
# -----------------------------------

def retrieve_transfers_sequencing(
        min_txn_size: int,
        training_end: str,
        epoch_reference_date: str = '',
        hybridize_wallet_ids: bool = False,
        dataset: str = 'prod',
    ) -> pd.DataFrame:
    """
    Returns buyer and seller sequence numbers for each wallet-coin pair, where the first
    buyer/seller receives rank 1. Only includes wallets from wallet_modeling_training_cohort.

    Note that first_buy will be null if there are no buys above the min_txn_size.

    Params:
    - min_txn_size (int): Minimum USD value to filter out dust/airdrops
    - training_end (str): Training period end as YYYY-MM-DD string
    - epoch_reference_date (str): Suffix added to table for each epoch
    - hybridize_wallet_ids (bool): Whether to use hybrid wallet-coin IDs vs regular wallet IDs
    - dataset (str): Set to 'prod' or 'dev' to alter query schema

    Returns:
    - sequence_df (DataFrame): Columns: wallet_address, coin_id, first_buy, first_sell,
        buyer_number, seller_number
    """
    # Identify which CTE to use
    if hybridize_wallet_ids:
        ordering_cte = 'hybridized_ordering'
    else:
        ordering_cte = 'base_ordering'

    # Identify which core schema to use
    if dataset == 'prod':
        schema = 'core'
    else:
        schema = 'dev_core'

    sequencing_sql = f"""
    with transaction_rank as (
        select coin_id
        ,wallet_address
        ,min(case when usd_net_transfers > 0 then date end) as first_buy
        ,min(case when usd_net_transfers < 0 then date end) as first_sell
        ,count(case when usd_net_transfers > 0 then date end) as buys_count
        ,count(case when usd_net_transfers < 0 then date end) as sells_count
        from {schema}.coin_wallet_profits cwp
        where abs(cwp.usd_net_transfers) >= {min_txn_size}
        and cwp.date <= '{training_end}'
        group by 1,2
    ),
    coin_ages as (
        select coin_id,
        date_diff(cast('{training_end}' as date), min(first_buy), DAY) as coin_age
        from transaction_rank
        group by 1
    ),
    wallet_ages as (
        select wallet_address,
        date_diff(cast('{training_end}' as date), min(first_buy), DAY) as wallet_age
        from transaction_rank
        group by 1
    ),
    buyer_ranks as (
        select coin_id
        ,wallet_address
        ,first_buy
        ,RANK() OVER (PARTITION BY coin_id ORDER BY first_buy ASC) as buyer_number
        from transaction_rank
        where first_buy is not null
    ),
    seller_ranks as (
        select coin_id
        ,wallet_address
        ,first_sell
        ,RANK() OVER (PARTITION BY coin_id ORDER BY first_sell ASC) as seller_number
        from transaction_rank
        where first_sell is not null
    ),
    sequence_ordering as (
        select tr.coin_id
        ,tr.wallet_address
        ,ca.coin_age
        ,wa.wallet_age
        ,tr.buys_count
        ,tr.sells_count
        ,tr.first_buy

        -- this handles 2 edge cases that create impossible hold times:
        ,case when (
            -- 1. ignore immaterial buys that precede a material sell
            tr.first_buy is not null
            -- 2. ignore the sequence that would generate a negative hold time:
            --     immaterial buy -> material sell -> material buy
            and tr.first_sell > tr.first_buy
        ) then tr.first_sell end as first_sell

        ,b.buyer_number
        ,s.seller_number
        from transaction_rank tr
        join coin_ages ca using (coin_id)
        join wallet_ages wa using (wallet_address)
        left join buyer_ranks b using (coin_id, wallet_address)
        left join seller_ranks s using (coin_id, wallet_address)
    ),

    ------ NON-HYBRID IDs CTE -------
    -- CTE for use when working with standard wallet ids
    base_ordering as (
        select
            so.*,
            wc.wallet_id as final_wallet_id
        from sequence_ordering so
        join (
            select wc.wallet_id,
            xw.wallet_address,
            from temp.wallet_modeling_training_cohort_{epoch_reference_date} wc
            join reference.wallet_ids xw on xw.wallet_id = wc.wallet_id
        ) wc using(wallet_address)
    ),


    ------ HYBRIDIZED IDs CTE -------
    -- CTE for use when working with hybridized ids
    hybridized_ordering as (
        select
            so.*,
            wc.hybrid_cw_id as final_wallet_id
        from sequence_ordering so
        join (
            select xw_cw.coin_id
            ,xw_cw.wallet_address
            ,xw_cw.hybrid_cw_id
            from temp.wallet_modeling_training_cohort_{epoch_reference_date} wc
            join reference.wallet_coin_ids xw_cw using(wallet_address)
        ) wc using(wallet_address,coin_id)
    )

    select
        final_wallet_id as wallet_address,
        coin_id,
        wallet_age,
        coin_age,
        buys_count,
        sells_count,
        first_buy,
        first_sell,
        buyer_number,
        seller_number
    from {ordering_cte}
    order by 1,2,3,4
    """
    try:
        sequence_df = bqu.run_query(sequencing_sql)
    except Exception as e:
        logger.error(f"FAILED SQL QUERY:\n{'=' * 80}\n{sequencing_sql}\n{'=' * 80}")
        logger.error(f"BigQuery Error: {str(e)}")
        raise RuntimeError(
            f"transfers_sequencing query failed. Original error: {str(e)}"
        ) from e
    logger.info("Retrieved sequence data for %s wallet-coin pairs across %s wallets",
                len(sequence_df), len(sequence_df['wallet_address'].unique()))

    # Data quality validation
    numeric_cols = ['wallet_age', 'coin_age', 'buys_count', 'sells_count', 'buyer_number', 'seller_number']
    for col in numeric_cols:
        if col in sequence_df.columns:
            negative_count = (sequence_df[col] < 0).sum()
            if negative_count > 0:
                logger.error(f"Found {negative_count} negative values in {col}")
                raise ValueError(f"Data quality check failed: {col} contains negative values")

    logger.info("Data quality validation passed - no negative values found in numeric columns")

    # Optimize memory usage
    sequence_df['coin_id'] = sequence_df['coin_id'].astype('category')
    sequence_df = u.df_downcast(sequence_df)

    return sequence_df
