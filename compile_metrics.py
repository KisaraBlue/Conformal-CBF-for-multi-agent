import os, sys, re
import numpy as np
import pandas as pd


if __name__ == "__main__":
    # Setup
    scene = sys.argv[1]
    forecaster = sys.argv[2]
    method = '' if len(sys.argv) < 4 else sys.argv[3]
    sigfigs=4

    if method == 'conformal CBF':
        if len(sys.argv) > 4:
            run_folder = sys.argv[4] + '/'
        else:
            run_folder = 'others/'
        metrics_folder = './metrics/' + scene + '/conformal_CBF/' + run_folder
    else:
        metrics_folder = './metrics/' + scene + '/decision_theory/'
    metrics_files = os.listdir(metrics_folder)
    metrics_dfs = [ pd.read_csv(os.path.join(metrics_folder, f)) for f in metrics_files if '.csv' in f ]
    metrics_df = pd.concat(metrics_dfs, ignore_index=True)
    if method == 'conformal CBF':
        metrics_df = metrics_df.drop(columns=['Unnamed: 0'], axis=1)
        metrics_df = metrics_df.sort_values(by=['QPr', 'al', 'dyn', 'acc', 'rep', 'att', 'sense', 'tau', 'loss', 'lr', 'eps'], ascending=[True, True, True, True, True, True, True, True, True, True, True])
        metrics_df.set_index(['QPr', 'al', 'dyn', 'acc', 'rep', 'att', 'sense', 'tau', 'loss', 'lr', 'eps'], inplace=True)
    else:
        metrics_df = metrics_df[metrics_df['forecaster'] == forecaster].drop(columns=['Unnamed: 0', 'forecaster'], axis=1)
        # Sort the metrics_df by first method, then lr
        metrics_df.loc[(metrics_df['method'] == 'conformal controller') & (metrics_df['lr'] == 0), 'method'] = 'Aggressive'
        metrics_df = metrics_df.sort_values(by=['method', 'lr'], ascending=[True, True])
        metrics_df.set_index(['method', 'lr'], inplace=True)

        # Find the columns with the character % and rename them to have a \% instead
        # (otherwise the LaTeX table will not compile)
        for col in metrics_df.columns:
            if '%' in col:
                metrics_df.rename(columns={col: col.replace('%', r'\%')}, inplace=True)

    # Function to round to two significant figures
    def round_to_sf(x, sf):
        if x == 0:
            return 0
        else:
            return round(x, sf - 1 - int(np.floor(np.log10(abs(x)))))

    # Function to apply the coloring and rounding
    def rounding(col):
        def format_value(x):
            if np.isinf(x):
                return '$\infty$'
            elif np.isnan(x):
                return 'NaN'
            else:
                x = round_to_sf(x, sigfigs)
                if np.isreal(x) and np.isclose(x, round(x)):
                    return str(int(round(x)))
                else:
                    return format(x, f".{sigfigs}g")
        return [format_value(x) if not isinstance(x, (bool, str)) else x for x in col]

    # Apply the function to each column
    styled_df = metrics_df.apply(rounding)

    if method == 'conformal CBF':

        csv_table = styled_df.to_csv()
        #csv_table = re.sub('solve_rate', 'QPrate', csv_table)
        #csv_table = re.sub('dynamics_type', 'dyn', csv_table)
        #csv_table = re.sub('K_', 'K', csv_table)
        #csv_table = re.sub('pred_rate', 'tau', csv_table)
        #csv_table = re.sub('loss_type', 'loss', csv_table)
        #csv_table = re.sub(r'goal\ time\ \(s\)', 'Tgoal', csv_table)
        #csv_table = re.sub(r'\ dist\ \(m\)', 'D', csv_table)
        csv_table = re.sub(',forecaster,method', '', csv_table)
        csv_table = re.sub(',NaN,NaN', '', csv_table)
        csv_table = re.sub('double_integral', 'DI', csv_table)
        #csv_table = re.sub('QPcontrol', 'QPval', csv_table)
        csv_table = re.sub('all_pred', 'allP', csv_table)
        csv_table = re.sub('gt_GT', 'gtGT', csv_table)
        csv_table = re.sub('no_learning', 'noLearn', csv_table)
        csv_table = re.sub(r'\.0,', ',', csv_table)
        #csv_table = re.sub(r'(\.\d*?)0+\s', r'\1 ', csv_table)
        #csv_table = re.sub(r'\.\s', ' ', csv_table)
        csv_table = re.sub('True', 'T', csv_table)
        csv_table = re.sub('False', 'F', csv_table)        

        output_folder = './metrics/' + scene + '/compiled_metrics_latex/'
        os.makedirs(output_folder, exist_ok=True)
        with open(output_folder + 'conformalCBF_' + run_folder.replace('/', '') + '.csv', 'w') as f:
            f.write(csv_table)
        
        latex_table = re.sub(r'.*50%D\n', '', csv_table)
        latex_table = re.sub(r',', ' & ', latex_table)
        latex_table = re.sub(r'\n', r'\\' + r'\\ ' + r'\\hline\n\t', latex_table)
        
        latex_doc = (
'''\\documentclass{article}
\\usepackage[landscape, left=0.5cm]{geometry}
\\usepackage{longtable}
\\begin{document}
\\begin{longtable}{|c|c|c|c|c|c|c|c|c|c|c|c|c|c|c|c|c|c|c|c|}
    \\hline
    $\\mathbf{r_{QP}}$ & $\\mathbf{\\alpha}$ & \\textbf{Dyn} & $\\mathbf{K_{acc}}$ & $\\mathbf{K_{rep}}$ & $\\mathbf{K_{att}}$ & $\\mathbf{\\rho_0}$ & $\\mathbf{\\tau}$ & \\textbf{Loss} & $\\mathbf{\\eta}$ & $\\mathbf{\\epsilon}$ & {\\small\\textbf{Reach}} & $\\mathbf{t_{goal}}$ & \\textbf{Safe} & {\\small$\\mathbf{n_{unsafe}}$} & $\\mathbf{\\%_{unsat}}$ & $\\mathbf{d_{min}}$ & $\\mathbf{d_{avg}}$ & $\\mathbf{d_{5\\%}}$ & $\\mathbf{d_{50\\%}}$ \\\\ \\hline
    \\endhead
    '''
+
latex_table
+
'''
\\end{longtable}
\\end{document}
'''
        )

        with open(output_folder + 'conformalCBF_' + run_folder.replace('/', '') + '.tex', 'w') as f:
            f.write(latex_doc)


    else:

        # Create a Styler object from the styled DataFrame
        styler = styled_df.style

        # Convert the Styler to a LaTeX table and save it to a file
        latex_table = styler.to_latex()

        # Remove unnecessary zeros after the decimal point
        latex_table = re.sub(r'(\.\d*?)0+\s', r'\1 ', latex_table)

        # If the above regex leaves a dot at the end of a number, remove the dot as well
        latex_table = re.sub(r'\.\s', ' ', latex_table)
        latex_table = re.sub(r'True', r'\\cmark', latex_table)
        latex_table = re.sub(r'False', r'\\xmark', latex_table)
        latex_table = re.sub(r'aci', r'\\makecell{ACI \\\\($\\alpha=0.01$)}', latex_table)
        latex_table = re.sub(r'conformal controller', r'\\makecell{Conformal \\\\Controller \\\\($\\epsilon=2$m)}', latex_table)
        latex_table = re.sub(r'conservative', r'Conservative', latex_table)

        # Save the LaTeX table to a file
        with open(os.path.join(metrics_folder, scene + '_' + forecaster + '.tex'), 'w') as f:
            f.write(latex_table)
