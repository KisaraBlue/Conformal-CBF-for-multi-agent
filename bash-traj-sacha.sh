#coupa_0 gates_2 hyang_1 hyang_8 little_1 little_3 nexus_6 quad_1 quad_3 coupa_1 hyang_0 hyang_3 little_0 little_2 nexus_5 quad_0 quad_2
for i in nexus_4; do
#$(find ./params/*); do
    for forecaster in darts; do
        filename=$(basename "$i")
        scene=$(echo "$filename" | rev | cut -d. -f2- | rev)
        #test_lr=1; test_eps=0; test_tau=4; test_al=1;
        solve_rate=4
        alpha=0.1
        Krep=10
        Katt=1
        rho0=500
        Kacc=1
        python plan_trajectory.py $scene $forecaster conformal\ CBF $solve_rate $alpha velocity_dyn $Krep $Katt $rho0 gound_truth -no_learning; # -overwrite;
        python make_results.py $scene $forecaster conformal\ CBF $solve_rate $alpha velocity_dyn $Krep $Katt $rho0 gound_truth -no_learning -no_video;
        python plan_trajectory.py $scene $forecaster conformal\ CBF $solve_rate $alpha double_integral $Kacc $Krep $Katt $rho0 gound_truth -no_learning; # -overwrite;
        python make_results.py $scene $forecaster conformal\ CBF $solve_rate $alpha double_integral $Kacc $Krep $Katt $rho0 gound_truth -no_learning -no_video;
        ### using predictions every tau frames
        tau=4
        python plan_trajectory.py $scene $forecaster conformal\ CBF $solve_rate $alpha velocity_dyn $Krep $Katt $rho0 $tau -no_learning; # -overwrite;
        python make_results.py $scene $forecaster conformal\ CBF $solve_rate $alpha velocity_dyn $Krep $Katt $rho0 $tau -no_learning -no_video;
        python plan_trajectory.py $scene $forecaster conformal\ CBF $solve_rate $alpha double_integral $Kacc $Krep $Katt $rho0 $tau -no_learning; # -overwrite;
        python make_results.py $scene $forecaster conformal\ CBF $solve_rate $alpha double_integral $Kacc $Krep $Katt $rho0 $tau -no_learning -no_video;
        ### using learning with parameters eta and epsilon
        lr=1
        eps=0
        python plan_trajectory.py $scene $forecaster conformal\ CBF $solve_rate $alpha velocity_dyn $Krep $Katt $rho0 gound_truth $lr $eps; # -overwrite;
        python make_results.py $scene $forecaster conformal\ CBF $solve_rate $alpha velocity_dyn $Krep $Katt $rho0 gound_truth $lr $eps -no_video;
        python plan_trajectory.py $scene $forecaster conformal\ CBF $solve_rate $alpha double_integral $Kacc $Krep $Katt $rho0 gound_truth $lr $eps; # -overwrite;
        python make_results.py $scene $forecaster conformal\ CBF $solve_rate $alpha double_integral $Kacc $Krep $Katt $rho0 gound_truth $lr $eps -no_video;
        ### using predictions and learning with tau, eta and epsilon
        python plan_trajectory.py $scene $forecaster conformal\ CBF $solve_rate $alpha velocity_dyn $Krep $Katt $rho0 $tau $lr $eps; # -overwrite;
        python make_results.py $scene $forecaster conformal\ CBF $solve_rate $alpha velocity_dyn $Krep $Katt $rho0 $tau $lr $eps;
        python plan_trajectory.py $scene $forecaster conformal\ CBF $solve_rate $alpha double_integral $Kacc $Krep $Katt $rho0 $tau $lr $eps; # -overwrite;
        python make_results.py $scene $forecaster conformal\ CBF $solve_rate $alpha double_integral $Kacc $Krep $Katt $rho0 $tau $lr $eps;
        break
        for lr in 1 100; do
            for epsilon in -0.5 0.5; do
                for tau in 2 8; do
                    for alpha in 0.5 8; do
                        python plan_trajectory.py $scene $forecaster conformal\ CBF $lr $epsilon $tau $alpha -overwrite;
                        python make_results.py $scene $forecaster conformal\ CBF $lr $epsilon $tau $alpha -no_video;
                        python plan_trajectory.py $scene $forecaster conformal\ CBF $lr $epsilon $tau $alpha -all_predictions -overwrite;
                        python make_results.py $scene $forecaster conformal\ CBF $lr $epsilon $tau $alpha -all_predictions -no_video;
                    done
                done
            done
        done
    done
done