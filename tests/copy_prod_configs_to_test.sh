# copies all prod configs to the tests/test_config folder
# sh copy_prod_configs_to_test.sh

for file in ../config/*; do
    cp "$file" "test_config/test_$(basename "$file")"
done
echo "Configs successfully copied to tests/test_config."
