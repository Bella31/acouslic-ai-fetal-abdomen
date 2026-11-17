import os

import pandas as pd
from openpyxl.styles import Font

from part1_frame_classification.scripts.evaluate_frame_selection import calc_precision_recall_from_conf_df

import pandas as pd

# def merge_conf_matrix(conf: pd.DataFrame, merge_classes=("irrelevant", "suboptimal"),
#                                new_class="non_optimal") -> pd.DataFrame:
#     """
#     Merge a 3-class confusion matrix into 2 classes by unifying `merge_classes`
#     into `new_class`.
#     Returns a new confusion matrix as a DataFrame.
#     """
#
#
#     return out

import pandas as pd


def merge_conf_matrix(conf):
    """
    Merge 3-class confusion matrix to two class based on merge_classes
    """
    cm_opt_only = None
    cm_rel = None
    try:
        cm_opt_only = pd.DataFrame(
            data=[[0, 0],
                  [0, 0]],
            index=["not-optimal", "optimal"],  # True labels (rows)
            columns=["not-optimal", "optimal"]  # Predicted labels (columns)
        )
        cm_opt_only.loc['not-optimal','not-optimal'] = (conf.loc['irrelevant']['irrelevant'] +
                                                       conf.loc['irrelevant']['suboptimal'] +
                                                       conf.loc['suboptimal']['irrelevant'] +
                                                       conf.loc['suboptimal']['suboptimal'])
        cm_opt_only.loc['not-optimal','optimal'] = conf.loc['irrelevant']['optimal'] + conf.loc['suboptimal']['optimal']
        cm_opt_only.loc['optimal','not-optimal'] = conf.loc['optimal']['irrelevant'] + conf.loc['optimal']['suboptimal']
        cm_opt_only.loc['optimal','optimal'] = conf.loc['optimal']['optimal']

        cm_rel = pd.DataFrame(
            data=[[0, 0],
                  [0, 0]],
            index=["irrelevant", "relevant"],  # True labels (rows)
            columns=["irrelevant", "relevant"]  # Predicted labels (columns)
        )
        cm_rel.loc['relevant', 'relevant'] = (conf.loc['optimal']['optimal'] +
                                              conf.loc['optimal']['suboptimal'] +
                                              conf.loc['suboptimal']['optimal'] +
                                              conf.loc['suboptimal']['suboptimal'])
        cm_rel.loc['relevant', 'irrelevant'] = conf.loc['optimal']['irrelevant'] + conf.loc['suboptimal']['irrelevant']
        cm_rel.loc['irrelevant', 'relevant'] = conf.loc['irrelevant']['optimal'] + conf.loc['irrelevant']['suboptimal']
        cm_rel.loc['irrelevant', 'irrelevant'] = conf.loc['irrelevant']['irrelevant']
    except:
        print('confusion matrices were not merged properly')
    return cm_opt_only, cm_rel

def calc_total(top1_5_avg_df, conf_df, top1_5_avg_df_subopt, pr_avg_subopt):
    """
    Calculate averages for all metrics except confusion matrix for the optimal data
    """
    top1_5_total_df = top1_5_avg_df.groupby("name", as_index=False).mean()
    top1_5_total_df_subopt = top1_5_avg_df_subopt.groupby("name", as_index=False).mean()
    pr_total_subopt = pr_avg_subopt.groupby("name", as_index=False).mean()

    conf_total_dfs = {}
    pr_total = {}

    for alg_dfs in conf_df:
        dfs = []
        for df in conf_df[alg_dfs].values():
            d = df.copy()
            if "True" in d.columns:
                d = d.set_index("True")
            d = d.apply(pd.to_numeric, errors="coerce").fillna(0)
            dfs.append(d)
        cm_sum = pd.concat(dfs, axis=0)
        conf_total_dfs[alg_dfs] = cm_sum.groupby(level=0).sum()

        conf_matrix_opt_only, conf_matrix_opt_subopt = merge_conf_matrix(conf_total_dfs[alg_dfs])
        precision_opt_only, recall_opt_only, f1_opt_only = calc_precision_recall_from_conf_df(conf_matrix_opt_only)
        precision_opt_subopt, recall_opt_subopt, f1_opt_subopt = calc_precision_recall_from_conf_df(conf_matrix_opt_subopt)
        pr = {}
        pr['precision_opt_only']=precision_opt_only
        pr['recall_opt_only']=recall_opt_only
        pr['f1_opt_only'] = f1_opt_only
        pr['precision_opt_subopt']=precision_opt_subopt
        pr['recall_opt_subopt'] = recall_opt_subopt
        pr['f1_opt_subopt'] = f1_opt_subopt
        pr_total[alg_dfs] = pr

    return top1_5_total_df, pr_total, conf_total_dfs, top1_5_total_df_subopt, pr_total_subopt

def write_dfs_to_sheet(
    writer: pd.ExcelWriter,

    dfs: dict,                 # {title: DataFrame}
    sheet_name: str,
    startrow: int = 0,
    space: int = 2,
    bold_titles: bool = True,
) -> int:
    """
    Write multiple DataFrames to the same Excel sheet, each with a title above it.
    Works with engine="openpyxl".
    Returns the next available row (0-based).
    """
    wb = writer.book  # openpyxl Workbook
    if sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
    else:
        ws = wb.create_sheet(title=sheet_name)
    # Ensure it's visible (prevents IndexError on save in some edge cases)
    ws.sheet_state = "visible"
    # Register worksheet in pandas writer
    writer.sheets[sheet_name] = ws
    row = startrow
    # If empty dict, still write something so the sheet exists and is visible
    if not dfs:
        cell = ws.cell(row=row + 1, column=1, value="No tables to write (dfs is empty).")
        if bold_titles:
            cell.font = Font(bold=True)
        return row + 1
    for title, df in dfs.items():
        # Title (openpyxl uses 1-based row/col)
        cell = ws.cell(row=row + 1, column=1, value=str(title))
        if bold_titles:
            cell.font = Font(bold=True)
        # DataFrame below title
        df.to_excel(
            writer,
            sheet_name=sheet_name,
            startrow=row + 1,   # pandas is 0-based
            startcol=0,
            index=False,
        )
        # Advance pointer:
        # 1 row for title + (header + len(df)) rows for dataframe + space rows
        row += 1 + (len(df) + 1) + space
    return row


if __name__ == "__main__":
    """
    In results_info the details of the runs are indicated : folder id is the key, and the values are a list containing
     fold number, experiment description
    """
    networks_path = '/media/bella/8A1D-C0A6/Academy/Home_ultrasound/output/network'
    results_path = '/home/bella/Academy/results/Opt_cases/overall_summary.xlsx'
    sheets_names = ["top1-5", 'precision_recall', 'conf_matrix']
    top_column_names = ['wfss', 'wfss_top3', 'wfss_top5', 'accuracy', 'accuracy_top3', 'accuracy_top5', 'inference_time']
    top_column_names_subopt = ['accuracy', 'accuracy_top3', 'accuracy_top5', 'inference_time']
    pr_column_names = ['precision_opt_only', 'recall_opt_only', 'F1_opt_only',	'precision_opt_subopt',
                       'recall_opt_subopt', 'F1_opt_subopt']
    pr_column_names_subopt = ['precision_opt_subopt', 'recall_opt_subopt', 'F1_opt_subopt']
    conf_column_names = ['irrelevant', 'optimal',	'suboptimal']
    conf_column_names_subopt = ['irrelevant', 'suboptimal']
    results_info = {
                    101: [0, 'Opt-only Resnet'],
                    102: [1, 'Opt-only Resnet'],
                    103: [2, 'Opt-only Resnet'],
                    104: [3, 'Opt-only Resnet'],
                    105: [4, 'Opt-only Resnet'],
                    165: [0, 'Opt-only Efficientnet'],
                    166: [1, 'Opt-only Efficientnet'],
                    167: [2, 'Opt-only Efficientnet'],
                    168: [3, 'Opt-only Efficientnet'],
                    169: [4, 'Opt-only Efficientnet'],
                    155: [0, 'Opt-only Convnext'],
                    156: [1, 'Opt-only Convnext'],
                    157: [2, 'Opt-only Convnext'],
                    158: [3, 'Opt-only Convnext'],
                    159: [4, 'Opt-only Convnext'],
                    170: [0, 'Opt-only Densenet'],
                    171: [1, 'Opt-only Densenet'],
                    172: [2, 'Opt-only Densenet'],
                    173: [3, 'Opt-only Densenet'],
                    174: [4, 'Opt-only Densenet'],
                    226: [0, 'One-step Resnet'],
                    227: [1, 'One-step Resnet'],
                    228: [2, 'One-step Resnet'],
                    229: [3, 'One-step Resnet'],
                    230: [4, 'One-step Resnet'],
                    236: [0, 'One-step Efficientnet'],
                    237: [1, 'One-step Efficientnet'],
                    238: [2, 'One-step Efficientnet'],
                    239: [3, 'One-step Efficientnet'],
                    240: [4, 'One-step Efficientnet'],
                    131: [0, 'One-step Convnext'],
                    132: [1, 'One-step Convnext'],
                    133: [2, 'One-step Convnext'],
                    134: [3, 'One-step Convnext'],
                    135: [4, 'One-step Convnext'],
                    196: [0, 'One-step Densenet'],
                    197: [1, 'One-step Densenet'],
                    198: [2, 'One-step Densenet'],
                    199: [3, 'One-step Densenet'],
                    200: [4, 'One-step Densenet'],
                    231: [0, 'two-step Resnet'],
                    232: [1, 'two-step Resnet'],
                    233: [2, 'two-step Resnet'],
                    234: [3, 'two-step Resnet'],
                    235: [4, 'two-step Resnet'],
                    241: [0, 'two-step Efficientnet'],
                    242: [1, 'two-step Efficientnet'],
                    243: [2, 'two-step Efficientnet'],
                    244: [3, 'two-step Efficientnet'],
                    245: [4, 'two-step Efficientnet'],
                    160: [0, 'two-step Convnext'],
                    161: [1, 'two-step Convnext'],
                    162: [2, 'two-step Convnext'],
                    163: [3, 'two-step Convnext'],
                    164: [4, 'two-step Convnext'],
                    201: [0, 'two-step (ours)'],
                    202: [1, 'two-step (ours)'],
                    203: [2, 'two-step (ours)'],
                    204: [3, 'two-step (ours)'],
                    205: [4, 'two-step (ours)'],
                    }
    top1_5_avg_df = None
    pr_avg_df = None
    conf_avg_df = None
    top1_5_avg_df_subopt = None
    pr_avg_df_subopt = None
    conf_df_subopt = None

    top1_5_avg_df = pd.DataFrame(columns=['name', 'fold'] + top_column_names)
    pr_avg_df = pd.DataFrame(columns=['name', 'fold'] + pr_column_names)
    conf_dfs = {}
    top1_5_avg_df_subopt = pd.DataFrame(columns=['name', 'fold'] + top_column_names_subopt)
    pr_avg_df_subopt = pd.DataFrame(columns=['name', 'fold'] + pr_column_names_subopt)
    conf_dfs_subopt = {}

    for run_id in results_info:
        if type(run_id) is str and '_' in run_id:
            splitted_run_id = run_id.split('_')
            network_id = splitted_run_id[0]
            postfix = "_" + "_".join(splitted_run_id[1:])
        else:
            network_id = run_id
            postfix = ""
        ind = results_info[run_id][0]
        run_series = pd.Series({'name': results_info[run_id][1], 'fold': results_info[run_id][0]})
        #opt data update
        excel_path = os.path.join(networks_path, str(network_id), 'results', f"predictions_{network_id}{postfix}.xlsx")
        dfs = pd.read_excel(excel_path,sheet_name=sheets_names)
        top_avg = dfs["top1-5"][top_column_names].mean()
        top1_5_avg_df.loc[results_info[run_id][1] + '_' + str(results_info[run_id][0])] = run_series.combine_first(top_avg)
        if results_info[run_id][1] not in conf_dfs:
            conf_dfs[results_info[run_id][1]] = {}
        conf_dfs[results_info[run_id][1]][results_info[run_id][0]] = dfs["conf_matrix"]
        #subopt data update
        excel_path = os.path.join(networks_path, str(network_id), 'results', f"predictions_{network_id}_subopt{postfix}.xlsx")
        dfs = pd.read_excel(excel_path, sheet_name=sheets_names)
        top_avg_subopt = dfs["top1-5"][top_column_names_subopt].mean()
        pr_avg_subopt = dfs["precision_recall"][pr_column_names_subopt].mean()
        top1_5_avg_df_subopt.loc[results_info[run_id][1] +  '_' + str(results_info[run_id][0])] = run_series.combine_first(top_avg_subopt)
        pr_avg_df_subopt.loc[results_info[run_id][1] +  '_' + str(results_info[run_id][0])] = run_series.combine_first(pr_avg_subopt)

    with (pd.ExcelWriter(results_path, engine="openpyxl") as writer):
        top1_5_total_df, pr_total, conf_total_dfs, top1_5_total_df_subopt, pr_total_subopt =\
            calc_total(top1_5_avg_df, conf_dfs, top1_5_avg_df_subopt, pr_avg_df_subopt)
        try:
            #write total calculations
            top1_5_total_df.to_excel(writer, sheet_name="top1_5", index=False)
            pr_total_df = pd.DataFrame(pr_total).T
            pr_total_df.to_excel(writer, sheet_name="pr", index=True)
            top1_5_total_df_subopt.to_excel(writer, sheet_name="top1_5_subopt", index=False)
            pr_total_subopt.to_excel(writer, sheet_name="pr_subopt", index=True)
            write_dfs_to_sheet(writer, conf_total_dfs, "conf")

            #write original data
            top1_5_avg_df.to_excel(writer, sheet_name="top1_5_folds", index=False)
      #      pr_avg_df.to_excel(writer, sheet_name="pr_folds", index=False)
            top1_5_avg_df_subopt.to_excel(writer, sheet_name="top_folds_subopt", index=False)

        except Exception as e:
            print("calc_total failed:", e)
            raise








