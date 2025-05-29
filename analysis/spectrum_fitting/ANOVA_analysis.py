# Copyright (c) 2025 Kyle Lopin (Naresuan University) <kylel@nu.ac.th>

"""

"""

__author__ = "Kyle Vitautas Lopin"


# installed libraries
import numpy as np
import pandas as pd
import pingouin as pg

pd.set_option('display.width', 200)
pd.set_option('display.max_columns', None)


def print_pg_anova_table(sensor: str, fruit: str = None, target: str = None):
    """
    Performs and prints in LaTex an ANOVA analysis for a given sensor type.

    This function reads ANOVA results from a file, applies statistical analysis
    (ANOVA and pairwise tests), formats the results, and saves the filtered
    results to a CSV file while also printing formatted tables.

    Parameters
    ----------
    sensor : str
        The sensor type for which the ANOVA analysis is performed.

    Returns
    -------
    None
        The function prints formatted ANOVA results and pairwise comparison tables
        (in LaTeX format) and saves filtered ANOVA results to a CSV file.

    Notes
    -----
    - The input CSV file is expected to be named "ANOVA_<sensor>.csv" and located
      in the "ANOVA_data" directory.
    - The p-values in the ANOVA results are adjusted for multiple comparisons using
      the Bonferroni correction method.
    - Pairwise comparisons are skipped for interacting factors (those containing "*").

    Workflow
    --------
    1. Read ANOVA results from "ANOVA_<sensor>.csv".
    2. Perform ANOVA using the dependent variable 'Score' and factors:
       - 'Leaf'
       - 'Measurement Type'
       - 'Integration Time'
       - 'LED Current'
    3. Apply Bonferroni correction to the p-values and add a new column, 'p-corrected'.
    4. Format the ANOVA table:
       - Convert 'DF' to integers.
       - Round 'SS' and 'F' values to two decimal places.
    5. Print the formatted ANOVA table.
    6. Extract and print non-interacting components (factors without "*") as a LaTeX table.
    7. Filter the ANOVA table to include only rows where 'p-corrected' < 0.05.
    8. For each significant non-interacting factor, perform pairwise comparisons
       and print results in LaTeX format.
    9. Save the filtered ANOVA table to "ANOVA_table_<sensor>.csv" in the "ANOVA_data" directory.
    """

    def conditional_format(value):
        """
        Format a value based on its magnitude:
        - Values > 0.01 are formatted as decimals with 3 decimal places.
        - Values <= 0.01 are formatted in scientific notation with 2 decimal places.
        - Values smaller than NumPy's floating-point tolerance are displayed as '< ε',
          and the tolerance is printed.

        Parameters
        ----------
        value : float
            The value to be formatted.

        Returns
        -------
        str
            The formatted string representation of the value.
        """
        tolerance = np.finfo(float).eps  # Numerical tolerance for float64

        if abs(value) < tolerance:
            print(f"Value {value:.2e} is smaller than the tolerance (ε = {tolerance:.2e})")
            return r"$< \varepsilon$"  # LaTeX string for '< ε'
        elif value > 0.01:
            return "{:.3f}".format(value)  # Decimal format with 3 decimal places
        else:
            return "{:.2e}".format(value)  # Scientific notation
    print(f"anova for {sensor}")
    filename = f"ANOVA_data/ANOVA_{sensor}.csv"
    df = pd.read_csv(filename)

    between_columns = ['Fruit', 'Measurement Type', 'Integration Time', 'Target',
                       'LED Current', 'Regressor']
    if fruit:
        df = df[df['Fruit'] == fruit]
        between_columns.remove('Fruit')
    if target:
        df = df[df['Target'] == target]
        between_columns.remove('Target')
    if False:  # filter the AS7263 data, its super confusing
        if sensor == "as7263":
            df = df[df['Measurement Type'] == "absorbance"]
            between_columns = ['Leaf', 'Integration Time', 'LED Current']

    # Perform the ANOVA test
    aov = pg.anova(dv='Score',
                   between=between_columns,
                   data=df)

    # Apply Bonferroni correction to the p-values
    aov['p-corrected'] = pg.multicomp(aov['p-unc'], method='bonferroni')[1]

    # Filter rows where p-corrected > 0.05
    filtered_aov = aov[aov['p-corrected'] < 0.05]
    # format columns for latex
    aov['DF'] = pd.to_numeric(aov['DF'], errors='coerce').astype(int)
    print(aov)
    for column in ['SS', 'F', 'MS']:
        aov[column] = aov[column].apply(lambda x: f"{x:.2f}")  # 2 decimal places for SS

    print(aov)

    # Filter non-interacting components (rows without "*")
    non_interacting = aov[~aov['Source'].str.contains("\*")]
    for column in ["p-unc", "p-corrected"]:
        non_interacting[column] = non_interacting[column].apply(conditional_format)

    # Print LaTeX table
    latex_table = non_interacting.to_latex(index=False, float_format="{:.2e}".format,
                                           escape=False,
                                           caption="ANOVA results for non-interacting components.",
                                           label=f"tab:{sensor}_anova")
    print(latex_table)
    print(filtered_aov)

    # Run pairwise tests for significant variables
    pairwise_results = {}
    for factor in filtered_aov['Source']:
        if '*' in factor:
            continue
        print(f"\nRunning pairwise comparisons for: {factor}")
        posthoc = pg.pairwise_tests(data=df, dv='Score', between=factor, padjust='bonf')
        # padjust corrects p-values, so rename the column to reflect this
        posthoc = posthoc.rename(columns={'p-unc': 'p-corrected'})
        pairwise_results[factor] = posthoc
        #
        print(posthoc[["Contrast", "A", "B", "T", "p-corr", "BF10"]])
        # for column in ['p-unc', 'p-corr']:
        #     # Scientific notation
        #     posthoc[column] = posthoc[column].map(
        #         lambda x: "1.0" if x == 1 else f"$10^{{{int(np.log10(x))}}}$" if x > 0 else "$0$")
        # Format specific columns
        posthoc["T"] = posthoc["T"].map(
            "{:.2f}".format)  # Format 'T' with 2 decimal places
        for column in ["p-corrected"]:
            posthoc[column] = posthoc[column].apply(conditional_format)
        pw_latex = posthoc[["Contrast", "A", "B", "T", "p-corrected", "BF10"]].to_latex(
            index=False, float_format="{:.2e}".format, escape=False,
            caption=f"{sensor} {factor} Pairwise tests.",
            label=f"tab:{sensor}_{factor}_pairwise_tests")
        print(pw_latex)

    # Save the filtered ANOVA table to a new CSV file
    output_filename = f"ANOVA_data/ANOVA_table_{sensor}.csv"
    filtered_aov.to_csv(output_filename, index=False)


def print_best_condition(sensor: str, fruit: str, target: str) -> None:
    """
    Load an ANOVA CSV file for a specific sensor, then filter and print
    the best condition for a given fruit and target.

    Parameters:
    - sensor: str, sensor identifier (used to build filename)
    - fruit: str, e.g., 'tomato'
    - target: str, e.g., 'lycopene (FW)'
    """

    # Load the file (you will fill in the correct path/format)
    df = pd.read_csv(f"ANOVA_data/ANOVA_{sensor}.csv")  # 👈 Fill this in

    # Filter by fruit and target
    filtered = df[(df['Fruit'] == fruit) & (df['Target'] == target)]

    if filtered.empty:
        print(f"No data found for Sensor='{sensor}', Fruit='{fruit}', Target='{target}'")
        return

    # Group by remaining experimental factors
    group_cols = ['Regressor', 'Measurement Type', 'Integration Time', 'LED Current']
    grouped = (
        filtered.groupby(group_cols)['Score']
        .mean()
        .reset_index()
        .sort_values(by='Score', ascending=False)
    )

    print(f"\n✅ Best conditions for sensor '{sensor}', fruit '{fruit}', target '{target}':\n")
    print(grouped.head(20))

    # Best single condition
    best = grouped.iloc[0]
    print("\n🥇 Best Overall Condition:")
    print(best.to_string())


if __name__ == '__main__':
    print_best_condition("as7262", "tomato", "lycopene (FW)")
    # print_pg_anova_table("as7262", "tomato", "lycopene (FW)")
