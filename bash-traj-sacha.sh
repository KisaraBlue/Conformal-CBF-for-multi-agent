#coupa_0 gates_2 hyang_1 hyang_8 little_1 little_3 nexus_6 quad_1 quad_3 coupa_1 hyang_0 hyang_3 little_0 little_2 nexus_5 quad_0 quad_2
metrics_run=run8
overwrite=" -overwrite" # " " or " -overwrite"
for i in nexus_4; do
#$(find ./params/*); do
    for forecaster in darts; do
        filename=$(basename "$i")
        scene=$(echo "$filename" | rev | cut -d. -f2- | rev)
        for solve_rate in 4; do
            for alpha in 0.1 1; do
                for Krep in 15 20; do
                    for Katt in 1; do
                        for rho0 in 400 800; do
                            for Kacc in 2 4; do
                                python plan_trajectory.py $scene $forecaster conformal\ CBF $solve_rate $alpha double_integral $Kacc $Krep $Katt $rho0 gound_truth -no_learning $overwrite;
                                python make_results.py $scene $forecaster conformal\ CBF $solve_rate $alpha double_integral $Kacc $Krep $Katt $rho0 gound_truth -no_learning $metrics_run -no_video;
                                for tau in 8 10 12; do
                                    python plan_trajectory.py $scene $forecaster conformal\ CBF $solve_rate $alpha double_integral $Kacc $Krep $Katt $rho0 $tau -no_learning $overwrite;
                                    python make_results.py $scene $forecaster conformal\ CBF $solve_rate $alpha double_integral $Kacc $Krep $Katt $rho0 $tau -no_learning $metrics_run -no_video;
                                done
                                for losstype in all_pred; do #gt_GT
                                    for lr in 10 100; do
                                        for eps in 0 0.2 0.4; do
                                            python plan_trajectory.py $scene $forecaster conformal\ CBF $solve_rate $alpha double_integral $Kacc $Krep $Katt $rho0 gound_truth $losstype $lr $eps $overwrite;
                                            python make_results.py $scene $forecaster conformal\ CBF $solve_rate $alpha double_integral $Kacc $Krep $Katt $rho0 gound_truth $losstype $lr $eps $metrics_run -no_video;
                                            for tau in 8 10 12; do
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

        break
        #for solve_rate in 4 8; do
        #    for alpha in 0.1 0.01; do
        #        for Krep in 5 25; do
        #            for Katt in 1 0.1; do
        #                for rho0 in 500 250; do
        #                    for Kacc in 1 0.1; do

        solve_rate=4
        alpha=0.1
        Krep=10
        Katt=1
        rho0=500
        Kacc=1
        python plan_trajectory.py $scene $forecaster conformal\ CBF $solve_rate $alpha velocity_dyn $Krep $Katt $rho0 gound_truth -no_learning; # -overwrite;
        python make_results.py $scene $forecaster conformal\ CBF $solve_rate $alpha velocity_dyn $Krep $Katt $rho0 gound_truth -no_learning -no_video;
        
        ### using predictions every tau frames
        tau=4
        python plan_trajectory.py $scene $forecaster conformal\ CBF $solve_rate $alpha velocity_dyn $Krep $Katt $rho0 $tau -no_learning; # -overwrite;
        python make_results.py $scene $forecaster conformal\ CBF $solve_rate $alpha velocity_dyn $Krep $Katt $rho0 $tau -no_learning -no_video;
        
        ### using learning with parameters eta and epsilon
        lr=1
        eps=0
        python plan_trajectory.py $scene $forecaster conformal\ CBF $solve_rate $alpha velocity_dyn $Krep $Katt $rho0 gound_truth $lr $eps; # -overwrite;
        python make_results.py $scene $forecaster conformal\ CBF $solve_rate $alpha velocity_dyn $Krep $Katt $rho0 gound_truth $lr $eps -no_video;

        ### using predictions and learning with tau, eta and epsilon
        python plan_trajectory.py $scene $forecaster conformal\ CBF $solve_rate $alpha velocity_dyn $Krep $Katt $rho0 $tau $lr $eps; # -overwrite;
        python make_results.py $scene $forecaster conformal\ CBF $solve_rate $alpha velocity_dyn $Krep $Katt $rho0 $tau $lr $eps;
        
    done
done