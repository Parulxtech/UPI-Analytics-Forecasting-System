
import glob
import os

import pandas as pd

MONTHLY_FILE = "data/upi_monthly.csv"
DAILY_FILE = "data/upi_daily.csv"
P2P_FILE = "data/upi_p2pandp2m.csv"
APPS_FOLDER = "data/upi_apps"

APP_COLUMNS = [
    "sr_no",
    "app",
    "ci_volume_mn",
    "ci_value_cr",
    "b2c_volume_mn",
    "b2c_value_cr",
    "b2b_volume_mn",
    "b2b_value_cr",
    "total_volume_mn",
    "total_value_cr",
]

APP_NUMBER_COLUMNS = APP_COLUMNS[2:]

# Normalize different spellings to one consistent application name.
APP_NAME_FIX = {
    "phone pe": "PhonePe",
    "phonepe": "PhonePe",
    "whatsapp": "WhatsApp",
    "cheq": "CheQ",
    "jummp": "Jumpp",
    "jumpp": "Jumpp",
}


def clean_app_name(value):
    """Remove markers and normalize known application name variations."""
    name = str(value).replace("#", "").strip()

    # Case-insensitive matching also handles unexpected capitalization.
    return APP_NAME_FIX.get(name.lower(), name)


def load_monthly():
    """Read monthly UPI data, oldest month first."""
    df = pd.read_csv(MONTHLY_FILE, thousands=",")

    if len(df.columns) != 4:
        raise ValueError("Monthly file should have exactly 4 columns")

    df.columns = ["month", "banks_live", "volume_mn", "value_cr"]
    df["month"] = pd.to_datetime(df["month"], format="%b-%y")

    df = df.sort_values("month").reset_index(drop=True)
    return df


def load_daily():
    """Read daily UPI data, oldest day first."""
    df = pd.read_csv(DAILY_FILE, thousands=",")

    if len(df.columns) != 3:
        raise ValueError("Daily file should have exactly 3 columns")

    df.columns = ["date", "volume_mn", "value_cr"]
    df["date"] = pd.to_datetime(df["date"], format="%d-%b-%y")

    df = df.sort_values("date").reset_index(drop=True)
    return df


def load_p2p_p2m():
    """Read P2P and P2M data, oldest month first."""
    df = pd.read_csv(P2P_FILE, thousands=",")

    if len(df.columns) != 7:
        raise ValueError("P2P/P2M file should have exactly 7 columns")

    df.columns = [
        "month",
        "total_volume_mn",
        "total_value_cr",
        "p2p_volume_mn",
        "p2p_value_cr",
        "p2m_volume_mn",
        "p2m_value_cr",
    ]

    # Example: 26-Sep means September 2026.
    df["month"] = pd.to_datetime(df["month"], format="%y-%b")

    df = df.sort_values("month").reset_index(drop=True)
    return df


def load_apps():
    """Read monthly app files and combine them into one clean table."""
    pattern = os.path.join(APPS_FOLDER, "upi_application_*.csv")
    files = sorted(glob.glob(pattern))

    if not files:
        raise FileNotFoundError("No app files found in " + APPS_FOLDER)

    frames = []

    for path in files:
        file_name = os.path.basename(path)

        # Example: upi_application_sep_26.csv -> Sep 2026.
        month_text = file_name.replace(
            "upi_application_", ""
        ).replace(".csv", "")

        month = pd.to_datetime(month_text, format="%b_%y")

        # Detect whether the file uses tabs or commas.
        with open(path, encoding="utf-8-sig") as file:
            first_line = file.readline()

        separator = "\t" if "\t" in first_line else ","

        # Read as text first to preserve and inspect original values.
        df = pd.read_csv(
            path,
            sep=separator,
            dtype=str,
            encoding="utf-8-sig",
        )

        if len(df.columns) != 10:
            raise ValueError(
                file_name + " should have exactly 10 columns"
            )

        df.columns = APP_COLUMNS
        df = df.drop(columns="sr_no")

        # Ignore rows with no application name.
        blank = (
            df["app"].isna()
            | df["app"].str.strip().eq("").fillna(False)
        )

        if blank.any():
            print(
                "INFO:",
                file_name,
                "has",
                int(blank.sum()),
                "row(s) without app name; ignored",
            )
            df = df.loc[~blank].copy()

        # Convert numeric columns.
        # A dash or blank means missing data, NOT zero.
        # Expected missing markers do not produce repeated warnings.
        for col in APP_NUMBER_COLUMNS:
            raw = df[col]

            cleaned_text = (
                raw.astype("string")
                .str.replace(",", "", regex=False)
                .str.strip()
            )

            missing_marker = (
                cleaned_text.isna()
                | cleaned_text.eq("").fillna(False)
                | cleaned_text.eq("-").fillna(False)
            )

            numeric_values = pd.to_numeric(
                cleaned_text.mask(missing_marker),
                errors="coerce",
            )

            # Warn only for unexpected, non-empty invalid values.
            bad = numeric_values.isna() & ~missing_marker

            for idx in df.index[bad]:
                print(
                    "WARNING:",
                    file_name,
                    "| app:",
                    df.loc[idx, "app"],
                    "| column:",
                    col,
                    "| unexpected non-numeric value:",
                    repr(raw.loc[idx]),
                )

            df[col] = numeric_values

        # Normalize app names after removing # and extra spaces.
        df["app"] = df["app"].apply(clean_app_name)

        df["month"] = month
        frames.append(df)

    data = pd.concat(frames, ignore_index=True)

    data = data.sort_values(
        ["month", "total_volume_mn"],
        ascending=[True, False],
        na_position="last",
    )

    return data.reset_index(drop=True)


def show_check(title, df, date_col, freq):
    """Print basic date and completeness checks."""
    full_range = pd.date_range(
        df[date_col].min(),
        df[date_col].max(),
        freq=freq,
    )

    print("===", title, "===")
    print("Rows:", len(df))
    print("First:", df[date_col].min().strftime("%d %b %Y"))
    print("Last:", df[date_col].max().strftime("%d %b %Y"))
    print("Duplicates:", df[date_col].duplicated().sum())
    print("Missing:", len(full_range) - df[date_col].nunique())


if __name__ == "__main__":
    monthly = load_monthly()
    daily = load_daily()
    p2p = load_p2p_p2m()
    apps = load_apps()

    show_check("MONTHLY", monthly, "month", "MS")
    show_check("DAILY", daily, "date", "D")
    show_check("P2P AND P2M", p2p, "month", "MS")

    # P2P + P2M should match Total, allowing small rounding differences.
    vol_gap = (
        p2p["p2p_volume_mn"]
        + p2p["p2m_volume_mn"]
        - p2p["total_volume_mn"]
    ).abs()

    val_gap = (
        p2p["p2p_value_cr"]
        + p2p["p2m_value_cr"]
        - p2p["total_value_cr"]
    ).abs()

    print(
        "Rows where P2P + P2M does not match Total (volume):",
        (vol_gap > 0.1).sum(),
    )
    print(
        "Rows where P2P + P2M does not match Total (value):",
        (val_gap > 0.1).sum(),
    )

    # Compare P2P/P2M totals with official monthly totals.
    both = p2p.merge(monthly, on="month")

    print("Months found in both files:", len(both))
    print(
        "Max volume difference vs monthly file:",
        (both["total_volume_mn"] - both["volume_mn"]).abs().max(),
    )
    print(
        "Max value difference vs monthly file:",
        (both["total_value_cr"] - both["value_cr"]).abs().max(),
    )

    # Application data checks.
    print("=== APPS ===")
    print("Rows:", len(apps))
    print("Months (files) read:", apps["month"].nunique())
    print("First month:", apps["month"].min().strftime("%b %Y"))
    print("Last month:", apps["month"].max().strftime("%b %Y"))
    print(
        "Duplicate month and app rows:",
        apps.duplicated(["month", "app"]).sum(),
    )
    print("Number of different apps:", apps["app"].nunique())
    print(
        "Empty number cells (including missing markers):",
        int(apps[APP_NUMBER_COLUMNS].isna().sum().sum()),
    )

    # Check component totals, ignoring rows with missing components.
    component_columns = [
        "ci_volume_mn",
        "b2c_volume_mn",
        "b2b_volume_mn",
    ]

    volume_parts = apps[component_columns]
    volume_complete = volume_parts.notna().all(axis=1)

    app_sum = volume_parts.sum(axis=1, min_count=3)
    volume_total_available = apps["total_volume_mn"].notna()

    volume_check = volume_complete & volume_total_available
    volume_mismatch = (
        app_sum.sub(apps["total_volume_mn"]).abs() > 0.1
    )

    print(
        "Rows where available components do not match Total volume:",
        int((volume_check & volume_mismatch).sum()),
    )
    print(
        "Rows skipped in volume component check due to missing values:",
        int((~volume_check).sum()),
    )

    # List all normalized app names.
    print("All app names found:")
    print(sorted(apps["app"].unique()))

    # Compare listed application totals with the full monthly UPI totals.
    # These are different measures and are not expected to be identical.
    sums = apps.groupby("month")[
        ["total_volume_mn", "total_value_cr"]
    ].sum(min_count=1)

    monthly_totals = monthly.set_index("month")[
        ["volume_mn", "value_cr"]
    ]

    check = sums.join(monthly_totals, how="left")
    check["volume_gap"] = (
        check["total_volume_mn"] - check["volume_mn"]
    ).round(2)
    check["value_gap"] = (
        check["total_value_cr"] - check["value_cr"]
    ).round(2)

    print("Listed app totals vs full monthly UPI totals:")
    print(
        check[
            [
                "total_volume_mn",
                "volume_mn",
                "volume_gap",
                "total_value_cr",
                "value_cr",
                "value_gap",
            ]
        ]
    )

    # Inspect the normalized names and the months in which they occur.
    for keyword in ["lxme", "cheq", "jumpp"]:
        part = apps[
            apps["app"].str.lower().str.contains(keyword, na=False)
        ]

        print("App name check:", keyword)
        print(
            part.groupby("app")["month"].apply(
                lambda dates: ", ".join(
                    dates.dt.strftime("%b %y")
                )
            )
        )
