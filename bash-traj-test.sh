#nexus_4 - coupa_0 hyang_1 - bookstore_0 gates_1 hyang_5
metrics_run=others
overwrite=" -overwrite" # " " or " -overwrite"
for i in coupa_0 hyang_1; do
#$(find ./params/*); do
    for forecaster in darts; do
        filename=$(basename "$i")
        scene=$(echo "$filename" | rev | cut -d. -f2- | rev)
        for solve_rate in 4; do
            for alpha in 1; do
                for Krep in 20; do
                    for Katt in 1; do
                        for rho0 in 400; do
                            for Kacc in 2; do
                                python plan_trajectory.py $scene $forecaster conformal\ CBF $solve_rate $alpha double_integral $Kacc $Krep $Katt $rho0 gound_truth -no_learning $overwrite;
                                python make_results.py $scene $forecaster conformal\ CBF $solve_rate $alpha double_integral $Kacc $Krep $Katt $rho0 gound_truth -no_learning $metrics_run -no_video;
                                for tau in 8; do
                                    python plan_trajectory.py $scene $forecaster conformal\ CBF $solve_rate $alpha double_integral $Kacc $Krep $Katt $rho0 $tau -no_learning $overwrite;
                                    python make_results.py $scene $forecaster conformal\ CBF $solve_rate $alpha double_integral $Kacc $Krep $Katt $rho0 $tau -no_learning $metrics_run -no_video;
                                done
                                for losstype in all_pred; do #gt_GT
                                    for lr in 10; do
                                        for eps in 0; do
                                            python plan_trajectory.py $scene $forecaster conformal\ CBF $solve_rate $alpha double_integral $Kacc $Krep $Katt $rho0 gound_truth $losstype $lr $eps $overwrite;
                                            python make_results.py $scene $forecaster conformal\ CBF $solve_rate $alpha double_integral $Kacc $Krep $Katt $rho0 gound_truth $losstype $lr $eps $metrics_run -no_video;
                                            for tau in 8; do
                                                python plan_trajectory.py $scene $forecaster conformal\ CBF $solve_rate $alpha double_integral $Kacc $Krep $Katt $rho0 $tau $losstype $lr $eps $overwrite;
                                                python make_results.py $scene $forecaster conformal\ CBF $solve_rate $alpha double_integral $Kacc $Krep $Katt $rho0 $tau $losstype $lr $eps $metrics_run -no_video;
                                            done
                                        done
                                    done
                                done
                            done
                        done
                    done
                done
            done
        done        
    done
done