# copies all prod configs to the tests/test_config folder
# run from the data-science repo root folder

for file in config/*; do
    cp "$file" "tests/test_config/test_$(basename "$file")"
done
