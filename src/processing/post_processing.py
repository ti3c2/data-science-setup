import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_importance(df, models, top_n=50, height=0.2, width=6):
    fi = pd.DataFrame(index=df.columns, columns=[])
    for i, m in enumerate(models):
        fi[f"m_{i}"] = m.get_feature_importance()

    fi = fi.stack().reset_index().iloc[:, [0, 2]]  # .to_frame()
    fi.columns = ["feature", "importance"]

    cols_ord = list(
        fi.groupby("feature")["importance"].mean().sort_values(ascending=False).index
    )

    print(
        "Всего признаков {} Усреднее по {} моделям: ".format(len(cols_ord), len(models))
    )
    cols_ord = cols_ord[:top_n]

    fi = fi[fi["feature"].isin(cols_ord)]

    plt.figure(figsize=(width, len(cols_ord) * height))
    b = sns.boxplot(data=fi, y="feature", x="importance", orient="h", order=cols_ord)

    print("На график нанесено топ-{} признаков".format(top_n))
    return (
        fi.groupby(by=["feature"], as_index=False)["importance"]
        .mean()
        .sort_values(by="importance", ascending=False)
    )
