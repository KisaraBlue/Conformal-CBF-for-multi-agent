#nexus_4 - coupa_0 hyang_1 - bookstore_0 gates_1 hyang_5
for i in nexus_4; do
    for forecaster in darts; do
        filename=$(basename "$i")
        scene=$(echo "$filename" | rev | cut -d. -f2- | rev)
        python compile_metrics.py $scene $forecaster conformal\ CBF run7
    done
done