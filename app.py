# ============================================================
# UPI ANALYTICS & FORECASTING SYSTEM
# Main Application - Part 1
# ============================================================

import os
import traceback
from datetime import datetime

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.linear_model import LinearRegression

# Authentication and database functions
from database import (
    initialize_db,
    create_user,
    authenticate_user,
    log_activity,
    get_all_users,
    get_activity_logs,
    set_user_active,
    change_password,
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="UPI Analytics & Forecasting System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown("""
<style>
/* Main app background */
.stApp {
    background-color: #0E1117 !important;
    color: #F8FAFC !important;
}

/* Default text */
.stApp p,
.stApp li,
.stApp label,
.stApp .stMarkdown {
    color: #E2E8F0;
}

/* Headings */
.stApp h1,
.stApp h2,
.stApp h3,
.stApp h4 {
    color: #FFFFFF !important;
}

/* FIX: KPI metric boxes */
div[data-testid="stMetric"] {
    background-color: #1E293B !important;
    border: 1px solid #475569 !important;
    border-radius: 12px !important;
    padding: 18px !important;
}

/* KPI labels */
div[data-testid="stMetric"] label,
div[data-testid="stMetric"] label p {
    color: #CBD5E1 !important;
}

/* KPI values */
div[data-testid="stMetric"] [data-testid="stMetricValue"],
div[data-testid="stMetric"] [data-testid="stMetricValue"] * {
    color: #FFFFFF !important;
}

/* Metric delta text */
div[data-testid="stMetric"] [data-testid="stMetricDelta"] {
    color: #E2E8F0 !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #20242E !important;
}

section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] label {
    color: #FFFFFF !important;
}

/* Download buttons and normal buttons */
.stButton button,
.stDownloadButton button {
    color: #FFFFFF !important;
    background-color: #2563EB !important;
    border: none !important;
}

/* Input fields */
.stTextInput input,
.stNumberInput input,
.stDateInput input,
.stSelectbox div[data-baseweb="select"] {
    color: #FFFFFF !important;
    background-color: #1E293B !important;
}

/* Text selection */
::selection {
    background-color: #2563EB;
    color: #FFFFFF;
}
/* Allow KPI values to use the available width */
div[data-testid="stMetric"] {
    min-width: 0 !important;
    width: 100% !important;
    padding: 12px !important;
    box-sizing: border-box !important;
}

div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    font-size: clamp(18px, 1.8vw, 30px) !important;
    white-space: normal !important;
    overflow: visible !important;
    text-overflow: clip !important;
    word-break: break-word !important;
    line-height: 1.3 !important;
}
</style>
""", unsafe_allow_html=True)



# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
        .main {
            background-color: #f7f9fc;
        }

        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
        }

        .main-title {
            font-size: 2rem;
            font-weight: 700;
            color: #17365d;
        }

        .subtitle {
            color: #5b6573;
            font-size: 1rem;
        }

        div[data-testid="stMetric"] {
            background-color: white;
            padding: 16px;
            border-radius: 12px;
            border: 1px solid #e3e8ef;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")


# ============================================================
# DATABASE INITIALIZATION
# ============================================================

try:
    initialize_db()
except Exception as exc:
    st.error("Database initialization failed.")
    st.code(str(exc))
    st.stop()


# ============================================================
# GENERAL HELPER FUNCTIONS
# ============================================================

def show_title(title, subtitle=None):
    """Display a page title and optional subtitle."""

    st.markdown(
        f'<div class="main-title">{title}</div>',
        unsafe_allow_html=True,
    )

    if subtitle:
        st.markdown(
            f'<div class="subtitle">{subtitle}</div>',
            unsafe_allow_html=True,
        )

    st.write("")


def format_number(value, decimals=2):
    """Format a numeric value safely."""

    if pd.isna(value):
        return "N/A"

    return f"{value:,.{decimals}f}"


def to_numeric(series):
    """Convert a pandas series into numeric values."""

    cleaned = (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.strip()
    )

    cleaned = cleaned.replace(
        {
            "": None,
            "-": None,
            "—": None,
            "NA": None,
            "N/A": None,
            "nan": None,
            "None": None,
        }
    )

    return pd.to_numeric(cleaned, errors="coerce")


def add_csv_download(dataframe, filename, button_label="Download CSV"):
    """Provide a CSV download button."""

    if dataframe is None or dataframe.empty:
        st.info("There is no data available to download.")
        return

    csv_data = dataframe.to_csv(index=False).encode("utf-8-sig")

    st.download_button(
        label=f"⬇️ {button_label}",
        data=csv_data,
        file_name=filename,
        mime="text/csv",
        key=f"download_{filename}",
    )


def show_dataframe(dataframe, filename, height=350):
    """Display a dataframe and its download button."""

    if dataframe is None or dataframe.empty:
        st.info("No data is available for this selection.")
        return

    st.dataframe(
        dataframe,
        use_container_width=True,
        height=height,
    )

    add_csv_download(dataframe, filename)


def safe_date_column(dataframe, column):
    """Convert a column into dates if it exists."""

    if column in dataframe.columns:
        dataframe[column] = pd.to_datetime(
            dataframe[column],
            errors="coerce",
        )

    return dataframe


# ============================================================
# LOAD PROJECT DATA
# ============================================================

@st.cache_data(show_spinner=False)
def load_all_data():
    """
    Load the verified NPCI datasets through data_loader.py.

    Missing values remain missing.
    The application does not generate replacement transaction data.
    """

    from data_loader import (
        load_monthly,
        load_daily,
        load_p2p_p2m,
        load_apps,
    )

    monthly = load_monthly()
    daily = load_daily()
    p2p_p2m = load_p2p_p2m()
    apps = load_apps()

    # Ensure the returned objects are DataFrames.
    monthly = pd.DataFrame(monthly).copy()
    daily = pd.DataFrame(daily).copy()
    p2p_p2m = pd.DataFrame(p2p_p2m).copy()
    apps = pd.DataFrame(apps).copy()

    # Normalize date columns where present.
    for dataframe in [monthly, daily, p2p_p2m, apps]:
        for column in [
            "date",
            "day",
            "month",
            "Date",
            "Day",
            "Month",
        ]:
            if column in dataframe.columns:
                dataframe[column] = pd.to_datetime(
                    dataframe[column],
                    errors="coerce",
                )

    return monthly, daily, p2p_p2m, apps


def load_data_safely():
    """Load all datasets and display useful error information."""

    try:
        return load_all_data()

    except Exception as exc:
        st.error("The project data could not be loaded.")

        st.warning(
            "Check that data_loader.py and the required files "
            "inside the data folder are available."
        )

        with st.expander("Technical error details"):
            st.code(str(exc))
            st.code(traceback.format_exc())

        return None, None, None, None


# ============================================================
# SESSION STATE
# ============================================================

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "username" not in st.session_state:
    st.session_state.username = None

if "user_role" not in st.session_state:
    st.session_state.user_role = None

if "user_email" not in st.session_state:
    st.session_state.user_email = None

if "current_page" not in st.session_state:
    st.session_state.current_page = "UPI Overview"

# ============================================================
# PART 2: LOGIN AND REGISTRATION SYSTEM
# ============================================================

def login_screen():
    """Display login and registration forms."""

    st.markdown(
        """
        <div style="text-align:center; padding:20px 0 10px 0;">
            <h1 style="color:#17365d;">
                UPI Analytics & Forecasting System
            </h1>
            <p style="font-size:17px; color:#5b6573;">
                Analyze UPI transactions, explore trends,
                and forecast future activity.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    left_space, center, right_space = st.columns([1, 1.5, 1])

    with center:

        login_tab, register_tab = st.tabs(
            ["🔐 Login", "📝 Create Account"]
        )

        # ----------------------------------------------------
        # LOGIN FORM
        # ----------------------------------------------------

        with login_tab:

            st.subheader("Welcome Back")
            st.write("Log in to access your dashboard.")

            with st.form("login_form"):

                username = st.text_input(
                    "Username",
                    placeholder="Enter your username",
                    key="login_username",
                ).strip()

                password = st.text_input(
                    "Password",
                    type="password",
                    placeholder="Enter your password",
                    key="login_password",
                )

                login_button = st.form_submit_button(
                    "Login",
                    use_container_width=True,
                    type="primary",
                )

            if login_button:

                if not username or not password:

                    st.error(
                        "Please enter both username and password."
                    )

                else:

                    try:
                        user = authenticate_user(
                            username,
                            password,
                        )

                        if user:

                            st.session_state.logged_in = True
                            st.session_state.username = user["username"]
                            st.session_state.user_role = user["role"]
                            st.session_state.user_email = user["email"]
                            st.session_state.current_page = "UPI Overview"

                            log_activity(
                                user["username"],
                                "LOGIN",
                                "User logged in successfully.",
                            )

                            st.success("Login successful!")

                            st.rerun()

                        else:

                            st.error(
                                "Invalid username or password, "
                                "or the account is inactive."
                            )

                    except Exception as exc:

                        st.error("Unable to log in.")

                        with st.expander("Technical details"):
                            st.code(str(exc))

        # ----------------------------------------------------
        # REGISTRATION FORM
        # ----------------------------------------------------

        with register_tab:

            st.subheader("Create a New Account")

            st.write(
                "Fill in the details below to register."
            )

            with st.form("registration_form"):

                new_username = st.text_input(
                    "Username",
                    placeholder="Choose a username",
                    key="register_username",
                ).strip()

                new_email = st.text_input(
                    "Email Address",
                    placeholder="example@email.com",
                    key="register_email",
                ).strip().lower()

                new_password = st.text_input(
                    "Password",
                    type="password",
                    placeholder="Create a password",
                    key="register_password",
                )

                confirm_password = st.text_input(
                    "Confirm Password",
                    type="password",
                    placeholder="Enter your password again",
                    key="register_confirm_password",
                )

                register_button = st.form_submit_button(
                    "Create Account",
                    use_container_width=True,
                    type="primary",
                )

            if register_button:

                # Validate username.
                if len(new_username) < 3:

                    st.error(
                        "Username must contain at least "
                        "3 characters."
                    )

                # Validate email.
                elif (
                    "@" not in new_email
                    or "." not in new_email.split("@")[-1]
                    or " " in new_email
                ):

                    st.error(
                        "Please enter a valid email address."
                    )

                # Validate password length.
                elif len(new_password) < 8:

                    st.error(
                        "Password must contain at least "
                        "8 characters."
                    )

                # Confirm password.
                elif new_password != confirm_password:

                    st.error(
                        "Password and confirm password do not match."
                    )

                else:

                    try:

                        success, message = create_user(
                            username=new_username,
                            email=new_email,
                            password=new_password,
                            role="user",
                        )

                        if success:

                            st.success(
                                "Account created successfully! "
                                "Please use the Login tab to log in."
                            )

                        else:

                            st.error(message)

                    except Exception as exc:

                        st.error(
                            "Registration could not be completed."
                        )

                        with st.expander("Technical details"):
                            st.code(str(exc))

    st.divider()

    st.caption(
        "UPI Analytics & Forecasting System | "
        "Academic Project"
    )


# ============================================================
# END OF PART 2
# ============================================================

# ============================================================
# PART 3: SIDEBAR, NAVIGATION AND LOGOUT
# ============================================================

def render_sidebar():
    """Display sidebar navigation for authenticated users."""

    st.sidebar.title("📊 UPI ")
    st.sidebar.caption("Analytics & Forecasting System")

    st.sidebar.divider()

    
    st.sidebar.write("### 👤 User Profile")

    st.sidebar.write(
        f"**Username:** {st.session_state.username}"
    )

    st.sidebar.write(
        f"**Role:** {str(st.session_state.user_role).title()}"
    )

    if st.session_state.user_email:
        st.sidebar.caption(st.session_state.user_email)


    # Available dashboard pages.
    pages = [
        "UPI Overview",
        "Daily & Monthly Analysis",
        "Application Analysis",
        "Forecasting",
        "Custom Data Analysis",
        "About",
        "Change Password",
    ]

    # Admin-only page.
    if st.session_state.user_role == "admin":
        pages.append("Admin Panel")

    current_page = st.session_state.get(
        "current_page",
        "UPI Overview",
    )

    if current_page not in pages:
        current_page = "UPI Overview"

    selected_page = st.sidebar.radio(
        "🧭 Navigation",
        options=pages,
        index=pages.index(current_page),
        key="main_navigation",
    )

    st.session_state.current_page = selected_page

    st.sidebar.divider()

    st.sidebar.info(
        "Use the navigation menu to explore UPI data, "
        "charts, forecasting and custom analysis."
    )

    # Display logout button.
    if st.sidebar.button(
        "🚪 Logout",
        use_container_width=True,
        key="logout_button",
    ):

        try:
            log_activity(
                st.session_state.username,
                "LOGOUT",
                "User logged out.",
            )
        except Exception:
            pass

        st.session_state.logged_in = False
        st.session_state.username = None
        st.session_state.user_role = None
        st.session_state.user_email = None
        st.session_state.current_page = "UPI Overview"

        st.rerun()

    return selected_page


# ============================================================
# DATA STATUS COMPONENT
# ============================================================

def render_data_status():
    """Show whether project datasets are available."""

    st.sidebar.divider()
    st.sidebar.subheader("📁 Data Status")

    required_files = [
        os.path.join(DATA_DIR, "upi_monthly.csv"),
        os.path.join(DATA_DIR, "upi_daily.csv"),
        os.path.join(DATA_DIR, "upi_p2pandp2m.csv"),
    ]

    for file_path in required_files:

        filename = os.path.basename(file_path)

        if os.path.isfile(file_path):
            st.sidebar.success(f"✓ {filename}")
        else:
            st.sidebar.error(f"✗ {filename}")

    apps_folder = os.path.join(DATA_DIR, "upi_apps")

    if os.path.isdir(apps_folder):

        app_files = [
            filename
            for filename in os.listdir(apps_folder)
            if filename.lower().endswith(".csv")
        ]

        if app_files:
            st.sidebar.success(
                f"✓ UPI Apps: {len(app_files)} CSV files"
            )
        else:
            st.sidebar.warning(
                "No application CSV files found."
            )

    else:
        st.sidebar.error("✗ upi_apps folder missing")


# ============================================================
# PAGE HEADER
# ============================================================

def render_page_header():
    """Display the common header for dashboard pages."""

    st.markdown(
        """
        <div style="
            background: linear-gradient(90deg, #17365d, #286090);
            padding: 18px 22px;
            border-radius: 12px;
            color: white;
            margin-bottom: 20px;
        ">
            <h2 style="margin:0;">
                UPI Analytics & Forecasting System
            </h2>
            <p style="margin:6px 0 0 0;">
                Explore transaction trends and analytical insights.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# END OF PART 3
# ============================================================

# ============================================================
# PART 4: UPI OVERVIEW DASHBOARD
# ============================================================

def find_column(dataframe, possible_names):
    """Find a column using possible column names."""

    if dataframe is None or dataframe.empty:
        return None

    column_lookup = {
        str(column).strip().lower(): column
        for column in dataframe.columns
    }

    for name in possible_names:
        key = name.strip().lower()

        if key in column_lookup:
            return column_lookup[key]

    return None


def get_numeric_column(dataframe, possible_names):
    """Return a numeric version of the first matching column."""

    column = find_column(dataframe, possible_names)

    if column is None:
        return None, None

    return column, to_numeric(dataframe[column])


def get_date_column(dataframe):
    """Identify a date or month column."""

    return find_column(
        dataframe,
        [
            "date",
            "day",
            "month",
            "transaction_date",
            "transaction_month",
        ],
    )


def prepare_time_series(dataframe):
    """Prepare a dataframe with a valid date column."""

    if dataframe is None or dataframe.empty:
        return None, None

    date_column = get_date_column(dataframe)

    if date_column is None:
        return None, None

    result = dataframe.copy()

    result[date_column] = pd.to_datetime(
        result[date_column],
        errors="coerce",
    )

    result = result.dropna(subset=[date_column])

    if result.empty:
        return None, None

    return result.sort_values(date_column), date_column


def render_overview(monthly, daily, p2p_p2m):
    """Render the main UPI overview dashboard."""

    show_title(
        "📊 UPI Overview",
        "A high-level view of UPI transaction activity "
        "using the available NPCI datasets.",
    )

    if monthly is None or monthly.empty:
        st.error(
            "Monthly transaction data is unavailable. "
            "Check data_loader.py and the monthly CSV file."
        )
        return

    monthly_data, monthly_date_column = prepare_time_series(
        monthly
    )

    if monthly_data is None:
        st.error(
            "The monthly dataset does not contain a valid "
            "date or month column."
        )
        st.write("Available columns:", list(monthly.columns))
        return

    volume_column, volume_values = get_numeric_column(
        monthly_data,
        [
            "volume_mn",
            "volume_million",
            "volume",
            "transaction_volume",
            "transactions_mn",
            "total_volume",
        ],
    )

    value_column, value_values = get_numeric_column(
        monthly_data,
        [
            "value_cr",
            "value_crore",
            "value",
            "transaction_value",
            "total_value",
            "amount_cr",
        ],
    )

    if volume_column is None or value_column is None:
        st.error(
            "Could not identify the transaction volume or value "
            "columns in the monthly dataset."
        )

        st.write("Available columns:", list(monthly_data.columns))

        st.info(
            "The dashboard expects volume and value columns "
            "from your data_loader.py output."
        )
        return

    monthly_data["_volume_numeric"] = volume_values
    monthly_data["_value_numeric"] = value_values

    # Do not count missing values as zero.
    total_volume = monthly_data["_volume_numeric"].sum(
        min_count=1
    )

    total_value = monthly_data["_value_numeric"].sum(
        min_count=1
    )

    valid_volume = monthly_data["_volume_numeric"].dropna()
    valid_value = monthly_data["_value_numeric"].dropna()

    average_monthly_volume = (
        valid_volume.mean()
        if not valid_volume.empty
        else None
    )

    average_monthly_value = (
        valid_value.mean()
        if not valid_value.empty
        else None
    )

    # Average transaction value in rupees:
    # Value in crore rupees / volume in million transactions
    # multiplied by 10.
    if (
        pd.notna(total_volume)
        and pd.notna(total_value)
        and total_volume > 0
    ):
        average_transaction_value = (
            total_value / total_volume
        ) * 10
    else:
        average_transaction_value = None

    # Determine the latest available month.
    latest_row = monthly_data.iloc[-1]

    latest_volume = latest_row["_volume_numeric"]
    latest_value = latest_row["_value_numeric"]

    st.caption(
        "The totals below cover the available monthly dataset. "
        "They are not necessarily calendar-year totals."
    )

    # --------------------------------------------------------
    # KPI CARDS
    # --------------------------------------------------------

    kpi1, kpi2, kpi3, kpi4 = st.columns(
    [1.15, 1.15, 1, 1.1]
)

    with kpi1:
        st.metric(
            "Total Transaction Volume",
            (
                f"{format_number(total_volume)} Mn"
                if pd.notna(total_volume)
                else "N/A"
            ),
        )

    with kpi2:
        st.metric(
            "Total Transaction Value",
            (
                f"₹ {format_number(total_value)} Cr"
                if pd.notna(total_value)
                else "N/A"
            ),
        )

    with kpi3:
        st.metric(
            "Average Transaction Value",
            (
                f"₹ {format_number(average_transaction_value)}"
                if average_transaction_value is not None
                else "N/A"
            ),
            help=(
                "Calculated from total value in crore rupees "
                "and volume in million transactions."
            ),
        )

    with kpi4:
        st.metric(
            "Latest Month Volume",
            (
                f"{format_number(latest_volume)} Mn"
                if pd.notna(latest_volume)
                else "N/A"
            ),
        )

    st.divider()

    # --------------------------------------------------------
    # MONTHLY TRANSACTION TRENDS
    # --------------------------------------------------------

    st.subheader("📈 Monthly Transaction Trends")

    chart_data = monthly_data[
        [
            monthly_date_column,
            "_volume_numeric",
            "_value_numeric",
        ]
    ].copy()

    chart_data = chart_data.rename(
        columns={
            monthly_date_column: "Month",
            "_volume_numeric": "Volume (Million)",
            "_value_numeric": "Value (Crore Rupees)",
        }
    )

    chart_data = chart_data.sort_values("Month")

    col1, col2 = st.columns(2)

    with col1:

        volume_chart = px.line(
            chart_data,
            x="Month",
            y="Volume (Million)",
            markers=True,
            title="Monthly Transaction Volume",
        )

        volume_chart.update_layout(
            xaxis_title="Month",
            yaxis_title="Transactions (Million)",
            hovermode="x unified",
        )

        st.plotly_chart(
            volume_chart,
            use_container_width=True,
        )

    with col2:

        value_chart = px.line(
            chart_data,
            x="Month",
            y="Value (Crore Rupees)",
            markers=True,
            title="Monthly Transaction Value",
        )

        value_chart.update_layout(
            xaxis_title="Month",
            yaxis_title="Value (Crore Rupees)",
            hovermode="x unified",
        )

        st.plotly_chart(
            value_chart,
            use_container_width=True,
        )

    add_csv_download(
        chart_data,
        "upi_monthly_overview.csv",
        "Download Monthly Overview",
    )

    # --------------------------------------------------------
    # MONTHLY GROWTH
    # --------------------------------------------------------

    st.subheader("📊 Latest Available Month Growth")

    growth_data = monthly_data[
        [
            monthly_date_column,
            "_volume_numeric",
            "_value_numeric",
        ]
    ].copy()

    growth_data = growth_data.sort_values(
        monthly_date_column
    )

    growth_data = growth_data.dropna(
        subset=[monthly_date_column]
    )

    if len(growth_data) >= 2:

        previous_row = growth_data.iloc[-2]
        current_row = growth_data.iloc[-1]

        previous_volume = previous_row["_volume_numeric"]
        current_volume = current_row["_volume_numeric"]

        previous_value = previous_row["_value_numeric"]
        current_value = current_row["_value_numeric"]

        growth_col1, growth_col2 = st.columns(2)

        with growth_col1:

            if (
                pd.notna(previous_volume)
                and pd.notna(current_volume)
                and previous_volume != 0
            ):

                volume_growth = (
                    (current_volume - previous_volume)
                    / abs(previous_volume)
                ) * 100

                st.metric(
                    "Volume Growth",
                    f"{volume_growth:.2f}%",
                    help=(
                        "Change between the last two available "
                        "monthly observations."
                    ),
                )

            else:

                st.metric(
                    "Volume Growth",
                    "N/A",
                )

        with growth_col2:

            if (
                pd.notna(previous_value)
                and pd.notna(current_value)
                and previous_value != 0
            ):

                value_growth = (
                    (current_value - previous_value)
                    / abs(previous_value)
                ) * 100

                st.metric(
                    "Value Growth",
                    f"{value_growth:.2f}%",
                    help=(
                        "Change between the last two available "
                        "monthly observations."
                    ),
                )

            else:

                st.metric(
                    "Value Growth",
                    "N/A",
                )

        st.caption(
            "Growth compares the last two available observations. "
            "If the latest month is incomplete, its growth may "
            "not represent a full-month comparison."
        )

    else:

        st.info(
            "At least two monthly observations are required "
            "to calculate growth."
        )

    st.divider()

    # --------------------------------------------------------
    # DAILY TRANSACTION TREND
    # --------------------------------------------------------

    st.subheader("📅 Daily Transaction Trend")

    if daily is not None and not daily.empty:

        daily_data, daily_date_column = prepare_time_series(
            daily
        )

        if daily_data is not None:

            daily_volume_column, daily_volume_values = (
                get_numeric_column(
                    daily_data,
                    [
                        "volume",
                        "volume_mn",
                        "volume_million",
                        "transaction_volume",
                        "transactions",
                        "total_volume",
                    ],
                )
            )

            if daily_volume_column is not None:

                daily_chart_data = daily_data[
                    [daily_date_column]
                ].copy()

                daily_chart_data["Volume"] = (
                    daily_volume_values
                )

                daily_chart_data = daily_chart_data.dropna(
                    subset=["Volume"]
                )

                if not daily_chart_data.empty:

                    daily_chart = px.line(
                        daily_chart_data,
                        x=daily_date_column,
                        y="Volume",
                        title="Daily Transaction Volume",
                        markers=False,
                    )

                    daily_chart.update_layout(
                        xaxis_title="Date",
                        yaxis_title="Volume (source dataset units)",
                        hovermode="x unified",
                    )

                    st.plotly_chart(
                        daily_chart,
                        use_container_width=True,
                    )

                    add_csv_download(
                        daily_chart_data,
                        "upi_daily_overview.csv",
                        "Download Daily Overview",
                    )

                else:

                    st.info(
                        "No valid daily volume observations "
                        "are available."
                    )

            else:

                st.warning(
                    "A daily volume column could not be identified."
                )

        else:

            st.warning(
                "The daily dataset does not contain a valid date "
                "column."
            )

    else:

        st.info("Daily transaction data is unavailable.")

    st.divider()

    # --------------------------------------------------------
    # P2P VS P2M ANALYSIS
    # --------------------------------------------------------

    st.subheader("🔄 P2P vs P2M Transaction Analysis")

    if p2p_p2m is not None and not p2p_p2m.empty:

        p2p_data = p2p_p2m.copy()

        p2p_date_column = get_date_column(p2p_data)

        p2p_volume_column = find_column(
            p2p_data,
            [
                "p2p_volume",
                "p2p_volume_mn",
                "p2p",
                "p2p_vol",
            ],
        )

        p2m_volume_column = find_column(
            p2p_data,
            [
                "p2m_volume",
                "p2m_volume_mn",
                "p2m",
                "p2m_vol",
            ],
        )

        if p2p_volume_column and p2m_volume_column:

            p2p_data[p2p_volume_column] = to_numeric(
                p2p_data[p2p_volume_column]
            )

            p2p_data[p2m_volume_column] = to_numeric(
                p2p_data[p2m_volume_column]
            )

            p2p_total = p2p_data[p2p_volume_column].sum(
                min_count=1
            )

            p2m_total = p2p_data[p2m_volume_column].sum(
                min_count=1
            )

            comparison_col1, comparison_col2 = st.columns(2)

            with comparison_col1:

                st.metric(
                    "P2P Total Volume",
                    (
                        format_number(p2p_total)
                        if pd.notna(p2p_total)
                        else "N/A"
                    ),
                )

            with comparison_col2:

                st.metric(
                    "P2M Total Volume",
                    (
                        format_number(p2m_total)
                        if pd.notna(p2m_total)
                        else "N/A"
                    ),
                )

            if (
                pd.notna(p2p_total)
                and pd.notna(p2m_total)
            ):

                comparison_chart_data = pd.DataFrame(
                    {
                        "Transaction Type": ["P2P", "P2M"],
                        "Volume": [p2p_total, p2m_total],
                    }
                )

                comparison_chart = px.bar(
                    comparison_chart_data,
                    x="Transaction Type",
                    y="Volume",
                    title="P2P vs P2M Total Volume",
                    text_auto=".2s",
                )

                comparison_chart.update_layout(
                    xaxis_title="Transaction Type",
                    yaxis_title="Volume (source dataset units)",
                )

                st.plotly_chart(
                    comparison_chart,
                    use_container_width=True,
                )

            if p2p_date_column:

                p2p_data[p2p_date_column] = pd.to_datetime(
                    p2p_data[p2p_date_column],
                    errors="coerce",
                )

                time_data = p2p_data.dropna(
                    subset=[p2p_date_column]
                ).copy()

                if not time_data.empty:

                    time_data = time_data.sort_values(
                        p2p_date_column
                    )

                    time_chart_data = pd.DataFrame(
                        {
                            "Month": time_data[p2p_date_column],
                            "P2P Volume": time_data[
                                p2p_volume_column
                            ],
                            "P2M Volume": time_data[
                                p2m_volume_column
                            ],
                        }
                    )

                    time_chart_data = time_chart_data.melt(
                        id_vars="Month",
                        var_name="Transaction Type",
                        value_name="Volume",
                    )

                    time_chart = px.line(
                        time_chart_data,
                        x="Month",
                        y="Volume",
                        color="Transaction Type",
                        markers=True,
                        title="P2P and P2M Trends Over Time",
                    )

                    st.plotly_chart(
                        time_chart,
                        use_container_width=True,
                    )

            add_csv_download(
                p2p_data,
                "upi_p2p_p2m_overview.csv",
                "Download P2P/P2M Data",
            )

        else:

            st.warning(
                "Could not identify P2P and P2M volume columns. "
                "Available columns are shown below."
            )

            st.write(list(p2p_data.columns))

    else:

        st.info("P2P/P2M transaction data is unavailable.")

    st.divider()

    # --------------------------------------------------------
    # MONTHLY DATA TABLE
    # --------------------------------------------------------

    with st.expander("📋 View Monthly Source Data"):

        show_dataframe(
            monthly,
            "upi_monthly_source_data.csv",
            height=350,
        )

    st.caption(
        "Data source: the NPCI datasets collected for this "
        "academic project. Figures reflect the available data "
        "and may be subject to source rounding or methodology."
    )


# ============================================================
# END OF PART 4
# ============================================================

# ============================================================
# PART 5: DAILY AND MONTHLY ANALYSIS
# ============================================================

def render_daily_monthly_analysis(monthly, daily):
    """Display daily and monthly transaction analysis."""

    show_title(
        "📅 Daily & Monthly Analysis",
        "Explore transaction trends, filter available data, "
        "and download the selected records.",
    )

    daily_tab, monthly_tab = st.tabs(
        [
            "📆 Daily Analysis",
            "🗓️ Monthly Analysis",
        ]
    )

    # ========================================================
    # DAILY ANALYSIS
    # ========================================================

    with daily_tab:

        st.subheader("Daily Transaction Analysis")

        if daily is None or daily.empty:

            st.warning("Daily transaction data is unavailable.")

        else:

            daily_data, date_column = prepare_time_series(daily)

            if daily_data is None:

                st.error(
                    "A valid date column could not be identified "
                    "in the daily dataset."
                )

                st.write(
                    "Available columns:",
                    list(daily.columns),
                )

            else:

                volume_column = find_column(
                    daily_data,
                    [
                        "volume",
                        "volume_mn",
                        "volume_million",
                        "transaction_volume",
                        "transactions",
                        "total_volume",
                    ],
                )

                value_column = find_column(
                    daily_data,
                    [
                        "value",
                        "value_cr",
                        "value_crore",
                        "transaction_value",
                        "total_value",
                    ],
                )

                daily_data = daily_data.copy()

                if volume_column:

                    daily_data[volume_column] = to_numeric(
                        daily_data[volume_column]
                    )

                if value_column:

                    daily_data[value_column] = to_numeric(
                        daily_data[value_column]
                    )

                # Date filters.
                valid_dates = daily_data[date_column].dropna()

                if valid_dates.empty:

                    st.warning(
                        "No valid dates were found in the dataset."
                    )

                else:

                    min_date = valid_dates.min().date()
                    max_date = valid_dates.max().date()

                    selected_dates = st.date_input(
                        "Select daily date range",
                        value=(min_date, max_date),
                        min_value=min_date,
                        max_value=max_date,
                        key="daily_date_range",
                    )

                    if (
                        isinstance(selected_dates, (tuple, list))
                        and len(selected_dates) == 2
                    ):

                        start_date, end_date = selected_dates

                        filtered_daily = daily_data[
                            daily_data[date_column].dt.date.between(
                                start_date,
                                end_date,
                            )
                        ].copy()

                    else:

                        filtered_daily = daily_data.copy()

                    if filtered_daily.empty:

                        st.info(
                            "No records are available for the "
                            "selected date range."
                        )

                    else:

                        # Daily volume KPI.
                        if volume_column:

                            daily_volume_total = (
                                filtered_daily[volume_column].sum(
                                    min_count=1
                                )
                            )

                            daily_volume_mean = (
                                filtered_daily[volume_column].mean()
                            )

                            kpi1, kpi2 = st.columns(2)

                            with kpi1:

                                st.metric(
                                    "Total Volume in Selection",
                                    format_number(
                                        daily_volume_total
                                    )
                                    if pd.notna(daily_volume_total)
                                    else "N/A",
                                )

                            with kpi2:

                                st.metric(
                                    "Average Daily Volume",
                                    format_number(
                                        daily_volume_mean
                                    )
                                    if pd.notna(daily_volume_mean)
                                    else "N/A",
                                )

                        # Daily value KPI.
                        if value_column:

                            daily_value_total = (
                                filtered_daily[value_column].sum(
                                    min_count=1
                                )
                            )

                            st.metric(
                                "Total Transaction Value",
                                format_number(daily_value_total)
                                if pd.notna(daily_value_total)
                                else "N/A",
                            )

                        # Daily volume chart.
                        if volume_column:

                            volume_chart = px.line(
                                filtered_daily,
                                x=date_column,
                                y=volume_column,
                                markers=True,
                                title="Daily Transaction Volume",
                            )

                            volume_chart.update_layout(
                                xaxis_title="Date",
                                yaxis_title=(
                                    "Volume (source dataset units)"
                                ),
                                hovermode="x unified",
                            )

                            st.plotly_chart(
                                volume_chart,
                                use_container_width=True,
                            )

                        # Daily value chart.
                        if value_column:

                            value_chart = px.line(
                                filtered_daily,
                                x=date_column,
                                y=value_column,
                                markers=True,
                                title="Daily Transaction Value",
                            )

                            value_chart.update_layout(
                                xaxis_title="Date",
                                yaxis_title=(
                                    "Value (source dataset units)"
                                ),
                                hovermode="x unified",
                            )

                            st.plotly_chart(
                                value_chart,
                                use_container_width=True,
                            )

                        # Download filtered daily data.
                        show_dataframe(
                            filtered_daily,
                            "filtered_daily_transactions.csv",
                            height=350,
                        )

    # ========================================================
    # MONTHLY ANALYSIS
    # ========================================================

    with monthly_tab:

        st.subheader("Monthly Transaction Analysis")

        if monthly is None or monthly.empty:

            st.warning(
                "Monthly transaction data is unavailable."
            )

        else:

            monthly_data, date_column = prepare_time_series(
                monthly
            )

            if monthly_data is None:

                st.error(
                    "A valid date or month column could not "
                    "be identified in the monthly dataset."
                )

                st.write(
                    "Available columns:",
                    list(monthly.columns),
                )

            else:

                volume_column = find_column(
                    monthly_data,
                    [
                        "volume_mn",
                        "volume_million",
                        "volume",
                        "transaction_volume",
                        "transactions_mn",
                        "total_volume",
                    ],
                )

                value_column = find_column(
                    monthly_data,
                    [
                        "value_cr",
                        "value_crore",
                        "value",
                        "transaction_value",
                        "total_value",
                        "amount_cr",
                    ],
                )

                monthly_data = monthly_data.copy()

                if volume_column:

                    monthly_data[volume_column] = to_numeric(
                        monthly_data[volume_column]
                    )

                if value_column:

                    monthly_data[value_column] = to_numeric(
                        monthly_data[value_column]
                    )

                valid_dates = monthly_data[date_column].dropna()

                if valid_dates.empty:

                    st.warning(
                        "No valid monthly dates were found."
                    )

                else:

                    min_date = valid_dates.min().date()
                    max_date = valid_dates.max().date()

                    selected_dates = st.date_input(
                        "Select monthly date range",
                        value=(min_date, max_date),
                        min_value=min_date,
                        max_value=max_date,
                        key="monthly_date_range",
                    )

                    if (
                        isinstance(selected_dates, (tuple, list))
                        and len(selected_dates) == 2
                    ):

                        start_date, end_date = selected_dates

                        filtered_monthly = monthly_data[
                            monthly_data[date_column].dt.date.between(
                                start_date,
                                end_date,
                            )
                        ].copy()

                    else:

                        filtered_monthly = monthly_data.copy()

                    if filtered_monthly.empty:

                        st.info(
                            "No monthly records match the "
                            "selected date range."
                        )

                    else:

                        # Monthly volume and value KPIs.
                        kpi1, kpi2 = st.columns(2)

                        if volume_column:

                            volume_total = (
                                filtered_monthly[volume_column].sum(
                                    min_count=1
                                )
                            )

                            with kpi1:

                                st.metric(
                                    "Total Transaction Volume",
                                    format_number(volume_total)
                                    if pd.notna(volume_total)
                                    else "N/A",
                                )

                        if value_column:

                            value_total = (
                                filtered_monthly[value_column].sum(
                                    min_count=1
                                )
                            )

                            with kpi2:

                                st.metric(
                                    "Total Transaction Value",
                                    format_number(value_total)
                                    if pd.notna(value_total)
                                    else "N/A",
                                )

                        # Monthly volume trend.
                        if volume_column:

                            volume_chart = px.bar(
                                filtered_monthly,
                                x=date_column,
                                y=volume_column,
                                title="Monthly Transaction Volume",
                            )

                            volume_chart.update_layout(
                                xaxis_title="Month",
                                yaxis_title=(
                                    "Volume (source dataset units)"
                                ),
                            )

                            st.plotly_chart(
                                volume_chart,
                                use_container_width=True,
                            )

                        # Monthly value trend.
                        if value_column:

                            value_chart = px.line(
                                filtered_monthly,
                                x=date_column,
                                y=value_column,
                                markers=True,
                                title="Monthly Transaction Value",
                            )

                            value_chart.update_layout(
                                xaxis_title="Month",
                                yaxis_title=(
                                    "Value (source dataset units)"
                                ),
                            )

                            st.plotly_chart(
                                value_chart,
                                use_container_width=True,
                            )

                        # Monthly growth calculation.
                        st.subheader("Month-on-Month Growth")

                        growth_data = filtered_monthly.sort_values(
                            date_column
                        ).copy()

                        if volume_column:

                            previous_volume = (
                                growth_data[volume_column].shift(1)
                            )

                            growth_data["Volume Growth (%)"] = (
                                (
                                    growth_data[volume_column]
                                    - previous_volume
                                )
                                / previous_volume.abs().where(
                                    previous_volume != 0
                                )
                            ) * 100

                        if value_column:

                            previous_value = (
                                growth_data[value_column].shift(1)
                            )

                            growth_data["Value Growth (%)"] = (
                                (
                                    growth_data[value_column]
                                    - previous_value
                                )
                                / previous_value.abs().where(
                                    previous_value != 0
                                )
                            ) * 100

                        growth_columns = [
                            date_column,
                        ]

                        if volume_column:
                            growth_columns.append(
                                "Volume Growth (%)"
                            )

                        if value_column:
                            growth_columns.append(
                                "Value Growth (%)"
                            )

                        available_growth_columns = [
                            column
                            for column in growth_columns
                            if column in growth_data.columns
                        ]

                        st.dataframe(
                            growth_data[available_growth_columns],
                            use_container_width=True,
                        )

                        add_csv_download(
                            growth_data[available_growth_columns],
                            "monthly_growth_analysis.csv",
                            "Download Growth Analysis",
                        )

                        # Download the selected monthly records.
                        show_dataframe(
                            filtered_monthly,
                            "filtered_monthly_transactions.csv",
                            height=350,
                        )


# ============================================================
# END OF PART 5
# ============================================================

# ============================================================
# PART 6: UPI APPLICATION ANALYSIS
# ============================================================

def render_application_analysis(apps):
    """Analyze UPI application data supplied by data_loader.py."""

    show_title(
        "📱 Application Analysis",
        "Compare the available UPI applications using the "
        "collected NPCI application dataset.",
    )

    if apps is None or apps.empty:
        st.error(
            "Application data is unavailable. Check the "
            "UPI application CSV files and data_loader.py."
        )
        return

    app_data = apps.copy()

    st.info(
        "Important: These figures represent the applications "
        "listed in the supplied dataset. Their totals may differ "
        "from overall UPI transaction totals because of NPCI "
        "reporting rules, application coverage, and rounding."
    )

    # --------------------------------------------------------
    # IDENTIFY THE IMPORTANT COLUMNS
    # --------------------------------------------------------

    app_column = find_column(
        app_data,
        [
            "app",
            "app_name",
            "application",
            "application_name",
            "upi_app",
            "name",
        ],
    )

    date_column = get_date_column(app_data)

    volume_column = find_column(
        app_data,
        [
            "volume_mn",
            "volume",
            "ci_volume_mn",
            "transaction_volume",
            "volume_million",
            "transactions_mn",
        ],
    )

    value_column = find_column(
        app_data,
        [
            "value_cr",
            "value",
            "ci_value_cr",
            "transaction_value",
            "value_crore",
        ],
    )

    if app_column is None:
        st.error(
            "Could not identify the application-name column."
        )
        st.write("Available columns:", list(app_data.columns))
        return

    if volume_column is None and value_column is None:
        st.error(
            "Could not identify a transaction volume or value "
            "column in the application dataset."
        )
        st.write("Available columns:", list(app_data.columns))
        return

    # --------------------------------------------------------
    # NORMALIZE APPLICATION DATA
    # --------------------------------------------------------

    app_data[app_column] = (
        app_data[app_column]
        .astype("string")
        .str.strip()
    )

    app_data = app_data[
        app_data[app_column].notna()
        & (app_data[app_column] != "")
    ].copy()

    if date_column is not None:

        app_data[date_column] = pd.to_datetime(
            app_data[date_column],
            errors="coerce",
        )

    if volume_column is not None:

        app_data[volume_column] = to_numeric(
            app_data[volume_column]
        )

    if value_column is not None:

        app_data[value_column] = to_numeric(
            app_data[value_column]
        )

    # --------------------------------------------------------
    # FILTERS
    # --------------------------------------------------------

    st.subheader("🔎 Filter Application Data")

    filter_col1, filter_col2 = st.columns(2)

    with filter_col1:

        app_names = sorted(
            app_data[app_column]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        selected_apps = st.multiselect(
            "Select applications",
            options=app_names,
            default=app_names,
            key="application_filter",
        )

    filtered_apps = app_data[
        app_data[app_column].astype(str).isin(selected_apps)
    ].copy()

    if date_column is not None:

        valid_dates = filtered_apps[date_column].dropna()

        if not valid_dates.empty:

            min_date = valid_dates.min().date()
            max_date = valid_dates.max().date()

            with filter_col2:

                selected_dates = st.date_input(
                    "Select date range",
                    value=(min_date, max_date),
                    min_value=min_date,
                    max_value=max_date,
                    key="application_date_range",
                )

            if (
                isinstance(selected_dates, (tuple, list))
                and len(selected_dates) == 2
            ):

                start_date, end_date = selected_dates

                filtered_apps = filtered_apps[
                    filtered_apps[date_column].dt.date.between(
                        start_date,
                        end_date,
                    )
                ].copy()

    if filtered_apps.empty:
        st.warning(
            "No application records match the selected filters."
        )
        return

    # --------------------------------------------------------
    # SUMMARY KPIs
    # --------------------------------------------------------

    st.subheader("📊 Application Summary")

    kpi1, kpi2, kpi3 = st.columns(3)

    with kpi1:

        st.metric(
            "Applications in Selection",
            filtered_apps[app_column].nunique(),
        )

    with kpi2:

        if volume_column:

            selected_volume = filtered_apps[
                volume_column
            ].sum(min_count=1)

            st.metric(
                "Listed-App Volume",
                (
                    f"{format_number(selected_volume)} Mn"
                    if pd.notna(selected_volume)
                    else "N/A"
                ),
            )

        else:

            st.metric("Listed-App Volume", "N/A")

    with kpi3:

        if value_column:

            selected_value = filtered_apps[
                value_column
            ].sum(min_count=1)

            st.metric(
                "Listed-App Value",
                (
                    f"₹ {format_number(selected_value)} Cr"
                    if pd.notna(selected_value)
                    else "N/A"
                ),
            )

        else:

            st.metric("Listed-App Value", "N/A")

    st.caption(
        "Volume and value are summed only across the selected "
        "records. Confirm the source column units before "
        "interpreting these totals."
    )

    st.divider()

    # --------------------------------------------------------
    # APP-WISE RANKING
    # --------------------------------------------------------

    st.subheader("🏆 Application Ranking")

    ranking_metric = st.selectbox(
        "Rank applications by",
        options=[
            option
            for option, column in [
                ("Transaction Volume", volume_column),
                ("Transaction Value", value_column),
            ]
            if column is not None
        ],
        key="application_ranking_metric",
    )

    ranking_column = (
        volume_column
        if ranking_metric == "Transaction Volume"
        else value_column
    )

    ranking = (
        filtered_apps.groupby(
            app_column,
            dropna=True,
        )[ranking_column]
        .sum(min_count=1)
        .reset_index()
        .sort_values(
            ranking_column,
            ascending=False,
            na_position="last",
        )
    )

    ranking = ranking.rename(
        columns={
            app_column: "Application",
            ranking_column: ranking_metric,
        }
    )

    ranking.insert(
        0,
        "Rank",
        range(1, len(ranking) + 1),
    )

    show_dataframe(
        ranking,
        "upi_application_ranking.csv",
        height=350,
    )

    # --------------------------------------------------------
    # TOP APPLICATIONS CHART
    # --------------------------------------------------------

    st.subheader("📈 Top Applications")

    top_n = st.slider(
        "Number of applications to display",
        min_value=5,
        max_value=max(5, min(30, len(ranking))),
        value=min(10, max(5, len(ranking))),
        key="application_top_n",
    )

    top_apps = ranking.head(top_n).copy()

    if not top_apps.empty:

        ranking_chart = px.bar(
            top_apps,
            x=ranking_metric,
            y="Application",
            orientation="h",
            title=f"Top {len(top_apps)} Applications by {ranking_metric}",
        )

        ranking_chart.update_layout(
            yaxis={"categoryorder": "total ascending"},
            xaxis_title=ranking_metric,
            yaxis_title="Application",
        )

        st.plotly_chart(
            ranking_chart,
            use_container_width=True,
        )

        add_csv_download(
            top_apps,
            "upi_top_applications.csv",
            "Download Top Applications",
        )

    st.divider()

    # --------------------------------------------------------
    # MONTHLY APP TRENDS
    # --------------------------------------------------------

    st.subheader("🗓️ Monthly Application Trends")

    if date_column is not None:

        trend_column = st.selectbox(
            "Choose the trend metric",
            options=[
                option
                for option, column in [
                    ("Transaction Volume", volume_column),
                    ("Transaction Value", value_column),
                ]
                if column is not None
            ],
            key="application_trend_metric",
        )

        trend_value_column = (
            volume_column
            if trend_column == "Transaction Volume"
            else value_column
        )

        trend_data = filtered_apps.dropna(
            subset=[date_column]
        ).copy()

        if not trend_data.empty:

            trend_data["_month"] = (
                trend_data[date_column]
                .dt.to_period("M")
                .dt.to_timestamp()
            )

            monthly_app_trend = (
                trend_data.groupby(
                    ["_month", app_column],
                    dropna=True,
                )[trend_value_column]
                .sum(min_count=1)
                .reset_index()
            )

            monthly_app_trend = monthly_app_trend.rename(
                columns={
                    "_month": "Month",
                    app_column: "Application",
                    trend_value_column: trend_column,
                }
            )

            trend_chart = px.line(
                monthly_app_trend,
                x="Month",
                y=trend_column,
                color="Application",
                markers=True,
                title=f"Monthly {trend_column} by Application",
            )

            trend_chart.update_layout(
                xaxis_title="Month",
                yaxis_title=trend_column,
                hovermode="x unified",
            )

            st.plotly_chart(
                trend_chart,
                use_container_width=True,
            )

            show_dataframe(
                monthly_app_trend,
                "upi_monthly_application_trends.csv",
                height=350,
            )

        else:

            st.info(
                "No valid dates are available for monthly trends."
            )

    else:

        st.warning(
            "The application dataset does not contain a "
            "recognized date or month column."
        )

    st.divider()

    # --------------------------------------------------------
    # FULL FILTERED DATA
    # --------------------------------------------------------

    with st.expander("📋 View Filtered Application Records"):

        show_dataframe(
            filtered_apps,
            "upi_filtered_application_data.csv",
            height=400,
        )

    # --------------------------------------------------------
    # NPCI METHODOLOGY NOTES
    # --------------------------------------------------------

    with st.expander("ℹ️ Understanding NPCI Application Data"):

        st.markdown(
            """
            - The analysis uses the application records supplied
              through this project's data loader.
            - Listed-application totals may not equal overall UPI
              totals because reporting coverage and methodology
              can differ.
            - Some small applications may be grouped under
              an "Other Apps" category in the source.
            - The source may use payer-side PSP attribution and
              other reporting rules.
            - A missing value is not the same as zero.
            - Application names and reporting categories can
              change over time. Similar names should not be
              combined unless their identity is verified.
            - These charts describe the supplied historical data;
              they do not establish the market share of every
              UPI application.
            """
        )

    st.caption(
        "Source: NPCI application data collected for this project. "
        "Interpret results according to the source's reporting "
        "methodology and the available records."
    )


# ============================================================
# END OF PART 6
# ============================================================

# ============================================================
# PART 7: UPI FORECASTING SYSTEM
# ============================================================

def render_forecasting(monthly):
    """Forecast monthly UPI transaction volume and value."""

    show_title(
        "🔮 UPI Transaction Forecasting",
        "Estimate future monthly transaction activity using "
        "historical observations and Linear Regression.",
    )

    st.info(
        "Forecasts are estimates generated from historical data. "
        "They are not official NPCI predictions and are not "
        "guaranteed to match future transactions."
    )

    if monthly is None or monthly.empty:
        st.error("Monthly data is unavailable.")
        return

    # --------------------------------------------------------
    # PREPARE DATA
    # --------------------------------------------------------

    data, date_column = prepare_time_series(monthly)

    if data is None:
        st.error(
            "A valid date or month column could not be identified."
        )
        st.write("Available columns:", list(monthly.columns))
        return

    volume_column = find_column(
        data,
        [
            "volume_mn",
            "volume_million",
            "volume",
            "transaction_volume",
            "transactions_mn",
            "total_volume",
        ],
    )

    value_column = find_column(
        data,
        [
            "value_cr",
            "value_crore",
            "value",
            "transaction_value",
            "total_value",
            "amount_cr",
        ],
    )

    if volume_column is None and value_column is None:
        st.error(
            "Could not identify transaction volume or value columns."
        )
        st.write("Available columns:", list(data.columns))
        return

    data = data.copy()

    if volume_column:
        data[volume_column] = to_numeric(data[volume_column])

    if value_column:
        data[value_column] = to_numeric(data[value_column])

    data = data.dropna(subset=[date_column]).sort_values(
        date_column
    )

    if data.empty:
        st.error("No valid dated observations are available.")
        return

    # Forecasting assumes one observation per month.
    data["_month"] = (
        data[date_column]
        .dt.to_period("M")
        .dt.to_timestamp()
    )

    # Combine duplicate monthly dates if any exist.
    aggregation = {}

    if volume_column:
        aggregation[volume_column] = lambda series: series.sum(
            min_count=1
        )

    if value_column:
        aggregation[value_column] = lambda series: series.sum(
            min_count=1
        )

    monthly_data = (
        data.groupby("_month", as_index=False)
        .agg(aggregation)
        .sort_values("_month")
    )

    if len(monthly_data) < 24:
        st.warning(
            "At least 24 monthly observations are recommended "
            "for this forecasting feature."
        )
        return

    st.caption(
        f"Available monthly observations: {len(monthly_data)} | "
        f"First month: {monthly_data['_month'].min():%b %Y} | "
        f"Last month: {monthly_data['_month'].max():%b %Y}"
    )

    # --------------------------------------------------------
    # USER CONTROLS
    # --------------------------------------------------------

    st.subheader("⚙️ Forecast Settings")

    setting_col1, setting_col2 = st.columns(2)

    with setting_col1:

        forecast_metric = st.selectbox(
            "Choose what to forecast",
            options=[
                option
                for option, column in [
                    ("Transaction Volume", volume_column),
                    ("Transaction Value", value_column),
                ]
                if column is not None
            ],
            key="forecast_metric",
        )

    with setting_col2:

        forecast_horizon = st.slider(
            "Number of future months",
            min_value=1,
            max_value=12,
            value=3,
            key="forecast_horizon",
        )

    target_column = (
        volume_column
        if forecast_metric == "Transaction Volume"
        else value_column
    )

    target_unit = (
        "million transactions"
        if forecast_metric == "Transaction Volume"
        else "crore rupees"
    )

    # Keep only valid observations for the selected metric.
    model_data = monthly_data[
        ["_month", target_column]
    ].dropna().copy()

    model_data = model_data.sort_values("_month")

    if len(model_data) < 24:
        st.error(
            "At least 24 valid observations are required for "
            "the selected metric."
        )
        return

    # Use a sequential month index rather than calendar timestamps.
    model_data["_time_index"] = range(len(model_data))

    X = model_data[["_time_index"]]
    y = model_data[target_column]

    # --------------------------------------------------------
    # TRAIN THE MODEL
    # --------------------------------------------------------

    model = LinearRegression()
    model.fit(X, y)

    model_data["Fitted Value"] = model.predict(X)

    # Historical fitted values help visualize the model.
    fitted_chart_data = model_data[
        ["_month", target_column, "Fitted Value"]
    ].rename(
        columns={
            "_month": "Month",
            target_column: "Actual Value",
        }
    )

    # --------------------------------------------------------
    # GENERATE FUTURE FORECASTS
    # --------------------------------------------------------

    last_index = len(model_data) - 1

    future_indices = list(
        range(
            last_index + 1,
            last_index + forecast_horizon + 1,
        )
    )

    future_dates = pd.date_range(
        start=(
            model_data["_month"].max()
            + pd.offsets.MonthBegin(1)
        ),
        periods=forecast_horizon,
        freq="MS",
    )

    future_predictions = model.predict(
        pd.DataFrame(
            {"_time_index": future_indices}
        )
    )

    # Transaction volume/value should not be negative.
    # Preserve the raw prediction for transparency.
    forecast_results = pd.DataFrame(
        {
            "Month": future_dates,
            "Raw Model Prediction": future_predictions,
            "Forecast": [
                max(0, prediction)
                for prediction in future_predictions
            ],
        }
    )

    forecast_results["Forecast Method"] = "Linear Regression"
    forecast_results["Forecast Metric"] = forecast_metric
    forecast_results["Unit"] = target_unit

    # --------------------------------------------------------
    # FORECAST SUMMARY
    # --------------------------------------------------------

    st.subheader("📌 Forecast Summary")

    latest_actual = model_data.iloc[-1][target_column]
    first_forecast = forecast_results.iloc[0]["Forecast"]
    last_forecast = forecast_results.iloc[-1]["Forecast"]

    summary_col1, summary_col2, summary_col3 = st.columns(3)

    with summary_col1:

        st.metric(
            "Latest Actual Observation",
            format_number(latest_actual),
        )

    with summary_col2:

        st.metric(
            "First Forecast",
            format_number(first_forecast),
        )

    with summary_col3:

        st.metric(
            "Final Forecast",
            format_number(last_forecast),
        )

    st.caption(
        f"Unit: {target_unit}. Forecasts are model estimates, "
        "not confirmed future transaction figures."
    )

    st.divider()

    # --------------------------------------------------------
    # ACTUAL AND FORECAST CHART
    # --------------------------------------------------------

    st.subheader("📈 Actual vs Forecast")

    actual_chart_data = model_data[
        ["_month", target_column]
    ].rename(
        columns={
            "_month": "Month",
            target_column: "Value",
        }
    )

    actual_chart_data["Series"] = "Historical Actual"

    forecast_chart_data = forecast_results[
        ["Month", "Forecast"]
    ].rename(
        columns={"Forecast": "Value"}
    )

    forecast_chart_data["Series"] = "Future Forecast"

    chart_data = pd.concat(
        [
            actual_chart_data,
            forecast_chart_data,
        ],
        ignore_index=True,
    )

    chart = px.line(
        chart_data,
        x="Month",
        y="Value",
        color="Series",
        markers=True,
        title=f"{forecast_metric}: Historical Data and Forecast",
    )

    chart.update_layout(
        xaxis_title="Month",
        yaxis_title=target_unit.title(),
        hovermode="x unified",
    )

    st.plotly_chart(
        chart,
        use_container_width=True,
    )

    # --------------------------------------------------------
    # MODEL FIT METRICS
    # --------------------------------------------------------

    st.subheader("🧪 Model Fit Information")

    fitted_values = model.predict(X)

    residuals = y.to_numpy() - fitted_values

    mae = float(
        abs(residuals).mean()
    )

    rmse = float(
        (residuals ** 2).mean() ** 0.5
    )

    metric_col1, metric_col2, metric_col3 = st.columns(3)

    with metric_col1:

        st.metric(
            "Mean Absolute Error (MAE)",
            format_number(mae),
        )

    with metric_col2:

        st.metric(
            "Root Mean Squared Error (RMSE)",
            format_number(rmse),
        )

    with metric_col3:

        st.metric(
            "Training R²",
            format_number(model.score(X, y)),
        )

    st.caption(
        "These are in-sample fit metrics. They do not measure "
        "how accurately the model will predict unseen future data. "
        "A time-based test set is needed for a stronger evaluation."
    )

    st.divider()

    # --------------------------------------------------------
    # DOWNLOAD FORECAST
    # --------------------------------------------------------

    st.subheader("📥 Forecast Results")

    show_dataframe(
        forecast_results,
        "upi_transaction_forecast.csv",
        height=350,
    )

    # --------------------------------------------------------
    # MODEL EXPLANATION
    # --------------------------------------------------------

    with st.expander("ℹ️ How does this forecast work?"):

        st.markdown(
            """
            **Model:** Linear Regression

            1. The model reads the available historical monthly data.
            2. Each observation is assigned a sequential time index.
            3. Linear Regression estimates a straight-line trend.
            4. The fitted model estimates the next 1–12 months.
            5. The forecast results can be downloaded as a CSV file.

            **Limitations**

            - The model assumes an approximately linear trend.
            - It does not automatically capture every seasonal effect.
            - Policy changes, new payment features, outages, and
              unusual economic events can affect future activity.
            - Forecast quality should be evaluated on historical
              observations excluded from model training.
            - Negative predictions are clipped to zero for display;
              the raw model prediction is also included in the results.
            """
        )

    st.caption(
        "Academic demonstration: forecasts are generated from "
        "the available historical dataset using Linear Regression."
    )


# ============================================================
# END OF PART 7
# ============================================================

# ============================================================
# PART 8: CUSTOM DATA ANALYSIS
# ============================================================

def render_custom_data_analysis():
    """Analyze a CSV file uploaded by the user."""

    show_title(
        "📁 Custom Data Analysis",
        "Upload your own CSV file to explore its structure, "
        "check data quality, create charts, and export results.",
    )

    st.info(
        "Your uploaded data is analyzed during this app session. "
        "This feature does not automatically add the uploaded "
        "file to the project's NPCI datasets."
    )

    # --------------------------------------------------------
    # CSV UPLOAD
    # --------------------------------------------------------

    uploaded_file = st.file_uploader(
        "Upload a CSV file",
        type=["csv"],
        help="Select a CSV file from your device.",
        key="custom_csv_upload",
    )

    if uploaded_file is None:

        st.write("### How to use this feature")

        st.markdown(
            """
            1. Click **Browse files**.
            2. Select a CSV file from your computer.
            3. Review the uploaded data and its quality report.
            4. Select numerical columns to create a chart.
            5. Download the analysis or cleaned data if needed.
            """
        )

        st.caption(
            "Tip: You can use a copy of your own dataset. "
            "Keep your original file unchanged."
        )

        return

    # --------------------------------------------------------
    # READ CSV SAFELY
    # --------------------------------------------------------

    try:

        uploaded_file.seek(0)

        custom_data = pd.read_csv(
            uploaded_file,
            encoding="utf-8-sig",
        )

    except UnicodeDecodeError:

        try:

            uploaded_file.seek(0)

            custom_data = pd.read_csv(
                uploaded_file,
                encoding="latin-1",
            )

        except Exception as exc:

            st.error("The CSV file could not be read.")
            st.code(str(exc))
            return

    except Exception as exc:

        st.error("The CSV file could not be read.")
        st.code(str(exc))
        return

    if custom_data.empty:

        st.warning("The uploaded CSV file contains no data.")
        return

    # Keep an unchanged copy for comparison.
    original_data = custom_data.copy(deep=True)

    st.success("CSV file uploaded and read successfully.")

    st.caption(
        f"File name: {uploaded_file.name}"
    )

    # --------------------------------------------------------
    # BASIC DATASET INFORMATION
    # --------------------------------------------------------

    st.subheader("📊 Dataset Overview")

    rows_count = len(original_data)
    columns_count = len(original_data.columns)
    duplicate_count = int(original_data.duplicated().sum())
    missing_cells = int(original_data.isna().sum().sum())

    info_col1, info_col2, info_col3, info_col4 = st.columns(4)

    with info_col1:
        st.metric("Rows", f"{rows_count:,}")

    with info_col2:
        st.metric("Columns", f"{columns_count:,}")

    with info_col3:
        st.metric("Duplicate Rows", f"{duplicate_count:,}")

    with info_col4:
        st.metric("Missing Cells", f"{missing_cells:,}")

    # --------------------------------------------------------
    # DATA PREVIEW
    # --------------------------------------------------------

    st.subheader("👀 Data Preview")

    preview_rows = st.slider(
        "Number of rows to preview",
        min_value=5,
        max_value=min(100, max(5, rows_count)),
        value=min(10, max(5, rows_count)),
        key="custom_preview_rows",
    )

    st.dataframe(
        original_data.head(preview_rows),
        use_container_width=True,
    )

    # --------------------------------------------------------
    # DATA TYPES AND MISSING VALUES
    # --------------------------------------------------------

    st.divider()
    st.subheader("🔍 Data Quality Report")

    quality_report = pd.DataFrame(
        {
            "Column": original_data.columns,
            "Data Type": [
                str(dtype)
                for dtype in original_data.dtypes
            ],
            "Missing Values": [
                int(value)
                for value in original_data.isna().sum().values
            ],
            "Missing Percentage": [
                round(
                    (value / len(original_data)) * 100,
                    2,
                )
                for value in original_data.isna().sum().values
            ],
            "Unique Values": [
                int(original_data[column].nunique(dropna=True))
                for column in original_data.columns
            ],
        }
    )

    show_dataframe(
        quality_report,
        "custom_data_quality_report.csv",
        height=300,
    )

    # Missing-value chart.
    missing_by_column = original_data.isna().sum()

    missing_by_column = (
        missing_by_column[missing_by_column > 0]
        .sort_values(ascending=False)
    )

    if not missing_by_column.empty:

        missing_chart_data = missing_by_column.rename(
            "Missing Values"
        ).reset_index()

        missing_chart_data.columns = [
            "Column",
            "Missing Values",
        ]

        missing_chart = px.bar(
            missing_chart_data,
            x="Column",
            y="Missing Values",
            title="Missing Values by Column",
        )

        st.plotly_chart(
            missing_chart,
            use_container_width=True,
        )

    else:

        st.success(
            "No missing values were found in the uploaded dataset."
        )

    # --------------------------------------------------------
    # DUPLICATE RECORDS
    # --------------------------------------------------------

    st.subheader("🔁 Duplicate Record Analysis")

    duplicate_rows = original_data[
        original_data.duplicated(keep=False)
    ].copy()

    if duplicate_rows.empty:

        st.success("No duplicate rows were found.")

    else:

        st.warning(
            f"{duplicate_count} duplicate rows were detected."
        )

        with st.expander("View duplicate records"):

            st.dataframe(
                duplicate_rows,
                use_container_width=True,
            )

            add_csv_download(
                duplicate_rows,
                "custom_duplicate_records.csv",
                "Download Duplicate Records",
            )

    # --------------------------------------------------------
    # NUMERICAL SUMMARY
    # --------------------------------------------------------

    st.divider()
    st.subheader("📐 Numerical Summary")

    numeric_columns = original_data.select_dtypes(
        include="number"
    ).columns.tolist()

    if numeric_columns:

        numeric_summary = original_data[
            numeric_columns
        ].describe().transpose().reset_index()

        numeric_summary = numeric_summary.rename(
            columns={"index": "Column"}
        )

        show_dataframe(
            numeric_summary,
            "custom_numeric_summary.csv",
            height=350,
        )

    else:

        st.info(
            "No numeric columns were detected. "
            "Check whether numerical values were imported "
            "as text."
        )

    # --------------------------------------------------------
    # OPTIONAL COLUMN CONVERSION
    # --------------------------------------------------------

    st.divider()
    st.subheader("🔢 Convert Text Columns to Numbers")

    st.write(
        "If a numerical column was imported as text, "
        "you can convert a copy of it for analysis."
    )

    text_columns = original_data.select_dtypes(
        include=["object", "string"]
    ).columns.tolist()

    if text_columns:

        selected_conversion_columns = st.multiselect(
            "Choose columns to convert",
            options=text_columns,
            default=[],
            key="custom_numeric_conversion",
        )

        converted_data = original_data.copy(deep=True)

        for column in selected_conversion_columns:

            converted_data[column] = to_numeric(
                converted_data[column]
            )

        if selected_conversion_columns:

            st.caption(
                "Unrecognized text values become missing values "
                "in this analysis copy. The original upload "
                "remains unchanged."
            )

            st.dataframe(
                converted_data.head(20),
                use_container_width=True,
            )

    else:

        converted_data = original_data.copy(deep=True)

        st.info(
            "There are no text columns available for conversion."
        )

    # --------------------------------------------------------
    # CREATE A CHART
    # --------------------------------------------------------

    st.divider()
    st.subheader("📈 Create Your Own Chart")

    chart_source = (
        converted_data
        if selected_conversion_columns
        else original_data
    )

    chart_columns = chart_source.columns.tolist()

    if len(chart_columns) >= 2:

        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:

            chart_type = st.selectbox(
                "Chart type",
                options=[
                    "Line Chart",
                    "Bar Chart",
                    "Scatter Plot",
                    "Histogram",
                ],
                key="custom_chart_type",
            )

        with chart_col2:

            x_column = st.selectbox(
                "X-axis / Column",
                options=chart_columns,
                key="custom_chart_x",
            )

        numeric_chart_columns = (
            chart_source.select_dtypes(include="number")
            .columns.tolist()
        )

        if chart_type == "Histogram":

            if numeric_chart_columns:

                histogram_column = st.selectbox(
                    "Numeric column",
                    options=numeric_chart_columns,
                    key="custom_histogram_column",
                )

                if st.button(
                    "Generate Chart",
                    key="custom_histogram_button",
                ):

                    histogram_data = chart_source[
                        [histogram_column]
                    ].dropna()

                    if not histogram_data.empty:

                        fig = px.histogram(
                            histogram_data,
                            x=histogram_column,
                            title=f"Distribution of {histogram_column}",
                        )

                        st.plotly_chart(
                            fig,
                            use_container_width=True,
                        )

                    else:

                        st.warning(
                            "The selected column has no valid values."
                        )

            else:

                st.warning(
                    "A histogram requires a numeric column."
                )

        else:

            if chart_type == "Scatter Plot":

                numeric_x_options = numeric_chart_columns

                if not numeric_x_options:

                    st.warning(
                        "A scatter plot requires numeric columns."
                    )

                    numeric_y_column = None

                else:

                    if x_column not in numeric_x_options:

                        x_column = st.selectbox(
                            "Choose a numeric X-axis",
                            options=numeric_x_options,
                            key="custom_scatter_x",
                        )

                    numeric_y_options = [
                        column
                        for column in numeric_chart_columns
                        if column != x_column
                    ]

                    if numeric_y_options:

                        numeric_y_column = st.selectbox(
                            "Y-axis",
                            options=numeric_y_options,
                            key="custom_scatter_y",
                        )

                    else:

                        numeric_y_column = None

            else:

                numeric_y_column = (
                    st.selectbox(
                        "Y-axis / Numeric value",
                        options=numeric_chart_columns,
                        key="custom_chart_y",
                    )
                    if numeric_chart_columns
                    else None
                )

            if st.button(
                "Generate Chart",
                key="custom_chart_generate",
            ):

                if numeric_y_column is None:

                    st.warning(
                        "Choose a suitable numeric column first."
                    )

                else:

                    chart_data = chart_source[
                        [x_column, numeric_y_column]
                    ].dropna()

                    if chart_data.empty:

                        st.warning(
                            "No valid records are available "
                            "for the selected columns."
                        )

                    else:

                        if chart_type == "Line Chart":

                            fig = px.line(
                                chart_data,
                                x=x_column,
                                y=numeric_y_column,
                                markers=True,
                                title=(
                                    f"{numeric_y_column} by {x_column}"
                                ),
                            )

                        elif chart_type == "Bar Chart":

                            fig = px.bar(
                                chart_data,
                                x=x_column,
                                y=numeric_y_column,
                                title=(
                                    f"{numeric_y_column} by {x_column}"
                                ),
                            )

                        else:

                            fig = px.scatter(
                                chart_data,
                                x=x_column,
                                y=numeric_y_column,
                                title=(
                                    f"{numeric_y_column} vs {x_column}"
                                ),
                            )

                        st.plotly_chart(
                            fig,
                            use_container_width=True,
                        )

                        add_csv_download(
                            chart_data,
                            "custom_chart_data.csv",
                            "Download Chart Data",
                        )

    else:

        st.info(
            "At least two columns are required to create "
            "a chart."
        )

    # --------------------------------------------------------
    # REMOVE DUPLICATES FROM A COPY
    # --------------------------------------------------------

    st.divider()
    st.subheader("🧹 Prepare a Cleaned Copy")

    st.write(
        "This option creates a separate copy without duplicate "
        "rows. It does not overwrite your original uploaded file."
    )

    remove_duplicates = st.checkbox(
        "Remove exact duplicate rows from the download copy",
        value=False,
        key="custom_remove_duplicates",
    )

    cleaned_data = original_data.copy(deep=True)

    if remove_duplicates:

        cleaned_data = cleaned_data.drop_duplicates().copy()

        st.success(
            f"Cleaned copy contains {len(cleaned_data):,} rows. "
            f"Removed {rows_count - len(cleaned_data):,} "
            "duplicate rows."
        )

    else:

        st.caption(
            "No duplicate rows will be removed unless "
            "you select the checkbox."
        )

    show_dataframe(
        cleaned_data,
        "custom_analyzed_data.csv",
        height=350,
    )

    st.success(
        "Analysis complete. Use the download buttons to export "
        "the report, chart data, duplicate records, or cleaned copy."
    )


# ============================================================
# END OF PART 8
# ============================================================

# ============================================================
# PART 9: ABOUT PAGE
# ============================================================

def render_about():
    """Display project information, features, methodology and FAQs."""

    show_title(
        "ℹ️ About UPI Analytics & Forecasting System",
        "Project information, features, data methodology, "
        "limitations and frequently asked questions.",
    )

    # --------------------------------------------------------
    # PROJECT INTRODUCTION
    # --------------------------------------------------------

    st.subheader("🎯 Project Introduction")

    st.write(
        """
        **UPI Analytics & Forecasting System** is a data analytics
        project developed to explore Unified Payments Interface
        (UPI) transaction data.

        The system helps users understand transaction trends,
        compare available payment application statistics,
        examine transaction patterns and generate experimental
        forecasts using historical data.

        The project combines Python, Streamlit, Pandas, Plotly
        and machine learning concepts in an interactive dashboard.
        """
    )

    # --------------------------------------------------------
    # PROJECT INFORMATION
    # --------------------------------------------------------

    st.subheader("📌 Project Information")

    project_col1, project_col2 = st.columns(2)

    with project_col1:

        st.markdown(
            """
            **Project Name**

            UPI Analytics & Forecasting System

            **Project Type**

            BCA Academic Project

            **Programming Language**

            Python

            **Interface**

            Streamlit
            """
        )

    with project_col2:

        st.markdown(
            """
            **Data Processing**

            Pandas and NumPy

            **Visualizations**

            Plotly

            **Forecasting**

            Linear Regression

            **Database**

            SQLite
            """
        )

    st.divider()

    # --------------------------------------------------------
    # OBJECTIVES
    # --------------------------------------------------------

    st.subheader("🎯 Project Objectives")

    objectives = [
        "Present historical UPI transaction statistics in an interactive dashboard.",
        "Analyze daily and monthly transaction trends.",
        "Explore available UPI application statistics.",
        "Compare available P2P and P2M transaction data.",
        "Generate experimental forecasts from historical monthly data.",
        "Allow users to upload and analyze their own CSV datasets.",
        "Provide download options for tables and analysis results.",
    ]

    for objective in objectives:
        st.markdown(f"- {objective}")

    st.divider()

    # --------------------------------------------------------
    # FEATURES
    # --------------------------------------------------------

    st.subheader("🚀 Main Features")

    features = [
        (
            "UPI Overview",
            "View key transaction indicators and available historical trends.",
        ),
        (
            "Daily & Monthly Analysis",
            "Explore transaction data over selected dates and months.",
        ),
        (
            "Application Analysis",
            "Compare statistics for applications included in the available dataset.",
        ),
        (
            "Forecasting",
            "Generate experimental forecasts using a simple Linear Regression model.",
        ),
        (
            "Custom Data Analysis",
            "Upload a CSV file, inspect missing values, identify duplicate rows and create charts.",
        ),
        (
            "User Authentication",
            "Register an account, sign in and change your password.",
        ),
        (
            "Admin Panel",
            "Allow authorized administrators to review user accounts and activity logs.",
        ),
        (
            "Data Export",
            "Download supported tables and analysis results as CSV files.",
        ),
    ]

    for feature_name, feature_description in features:

        with st.expander(feature_name):

            st.write(feature_description)

    st.divider()

    # --------------------------------------------------------
    # DATA SOURCES AND COVERAGE
    # --------------------------------------------------------

    st.subheader("📂 Data Sources and Coverage")

    st.write(
        """
        The project uses manually collected historical UPI statistics
        from published NPCI data, stored in local CSV files.

        The dataset files include monthly transaction statistics,
        daily transaction statistics, P2P/P2M statistics and
        available UPI application statistics.
        """
    )

    st.info(
        "The dashboard uses the data available in the project's "
        "CSV files. The data is not automatically refreshed from "
        "NPCI, so the latest figures may not be available."
    )

    st.markdown(
        """
        **Important data interpretation notes:**

        - Monthly and daily datasets may cover different time periods.
        - The latest month may be incomplete.
        - Application statistics represent the applications included
          in the available source data, not necessarily all UPI transactions.
        - Small differences between reported totals can occur because
          of rounding or differences in reporting scope.
        - A missing value means the value was not available in the source;
          it must not automatically be interpreted as zero.
        """
    )

    st.divider()

    # --------------------------------------------------------
    # METHODOLOGY
    # --------------------------------------------------------

    st.subheader("⚙️ Methodology")

    methodology = [
        (
            "1. Data Collection",
            "Historical statistics were manually collected from published NPCI data and stored in CSV files.",
        ),
        (
            "2. Data Loading",
            "Python and Pandas load the CSV files for analysis.",
        ),
        (
            "3. Data Validation",
            "The application checks available columns, dates, numeric values and missing data.",
        ),
        (
            "4. Data Analysis",
            "Summary statistics and comparisons are calculated from the available records.",
        ),
        (
            "5. Visualization",
            "Plotly charts present trends and comparisons interactively.",
        ),
        (
            "6. Forecasting",
            "Linear Regression uses historical observations to estimate future values.",
        ),
        (
            "7. Export",
            "Users can download supported analysis results for further study.",
        ),
    ]

    for step_title, step_description in methodology:

        with st.expander(step_title):

            st.write(step_description)

    st.warning(
        "Forecasts are estimates generated from historical patterns. "
        "They are not official NPCI projections and should not be "
        "treated as guaranteed future transaction figures."
    )

    st.divider()

    # --------------------------------------------------------
    # TECHNOLOGY STACK
    # --------------------------------------------------------

    st.subheader("💻 Technology Stack")

    technology_data = pd.DataFrame(
        {
            "Technology": [
                "Python",
                "Streamlit",
                "Pandas",
                "NumPy",
                "Plotly",
                "Scikit-learn",
                "SQLite",
                "bcrypt",
            ],
            "Purpose": [
                "Core programming language",
                "Interactive web application",
                "Data loading and analysis",
                "Numerical operations",
                "Interactive charts",
                "Linear Regression forecasting",
                "Local user database",
                "Password hashing",
            ],
        }
    )

    st.dataframe(
        technology_data,
        use_container_width=True,
        hide_index=True,
    )

    st.divider()

    # --------------------------------------------------------
    # LIMITATIONS
    # --------------------------------------------------------

    st.subheader("⚠️ Project Limitations")

    limitations = [
        "The project uses locally stored historical data and does not automatically fetch live NPCI statistics.",
        "Forecast accuracy depends on the quantity, quality and patterns of the historical data.",
        "Linear Regression may not capture sudden changes, seasonal effects or unusual events.",
        "Application-level totals may not equal the complete UPI transaction total because the reporting scope can differ.",
        "The project is an academic analytics tool and is not an official NPCI application.",
        "SQLite is suitable for a small academic project but may not be sufficient for a large production service.",
    ]

    for limitation in limitations:
        st.markdown(f"- {limitation}")

    st.divider()

    # --------------------------------------------------------
    # PRIVACY AND SECURITY
    # --------------------------------------------------------

    st.subheader("🔐 Privacy and Security")

    st.write(
        """
        User account information is stored in the project's local
        SQLite database. Passwords should be stored as password hashes
        rather than plain text.

        Access to administrative features is restricted by application
        roles. Activity logs help record supported account activities.
        """
    )

    st.warning(
        "This is an academic project. Before deploying it publicly, "
        "additional security measures should be implemented, including "
        "secure configuration, HTTPS, database protection, rate limiting "
        "and appropriate handling of personal information."
    )

    st.divider()

    # --------------------------------------------------------
    # FREQUENTLY ASKED QUESTIONS
    # --------------------------------------------------------

    st.subheader("❓ Frequently Asked Questions")

    with st.expander("1. Is this an official NPCI application?"):

        st.write(
            "No. This is an independent academic project that analyzes "
            "historical UPI statistics. It is not an official NPCI product."
        )

    with st.expander("2. Does the dashboard show live UPI transactions?"):

        st.write(
            "No. It analyzes the CSV data available in the project. "
            "It does not display live payment transactions."
        )

    with st.expander("3. How does the forecasting feature work?"):

        st.write(
            "The forecasting feature uses Linear Regression to estimate "
            "future values from historical monthly observations. "
            "The estimates are experimental and may differ from actual results."
        )

    with st.expander("4. Can I analyze my own data?"):

        st.write(
            "Yes. Open Custom Data Analysis and upload a supported CSV file. "
            "You can inspect the data, check missing values, create charts "
            "and download supported results."
        )

    with st.expander("5. Why can application totals differ from overall UPI totals?"):

        st.write(
            "The application dataset may have a different reporting scope "
            "from the overall transaction dataset. Rounding and source "
            "methodology can also affect comparisons."
        )

    with st.expander("6. Does a missing value mean zero transactions?"):

        st.write(
            "No. A missing value means that the value is unavailable "
            "in the dataset. It should not automatically be treated as zero."
        )

    with st.expander("7. Can users download analysis results?"):

        st.write(
            "Download options are provided on supported analysis pages. "
            "The available export depends on the selected feature."
        )

    # --------------------------------------------------------
    # FOOTER
    # --------------------------------------------------------

    st.divider()

    st.caption(
        "UPI Analytics & Forecasting System | "
        "BCA Academic Project | Built with Python and Streamlit"
    )


# ============================================================
# END OF PART 9
# ============================================================

# ============================================================
# PART 10: CHANGE PASSWORD
# ============================================================

def render_change_password():
    """Allow the logged-in user to change their password."""

    show_title(
        "🔐 Change Password",
        "Update your account password securely.",
    )

    username = st.session_state.get("username")

    if not username:
        st.error("Your session is not valid. Please log in again.")
        return

    st.info(
        "Enter your current password and choose a new password "
        "that is at least 8 characters long."
    )

    with st.form("change_password_form"):

        current_password = st.text_input(
            "Current Password",
            type="password",
        )

        new_password = st.text_input(
            "New Password",
            type="password",
            help="Use at least 8 characters.",
        )

        confirm_password = st.text_input(
            "Confirm New Password",
            type="password",
        )

        submitted = st.form_submit_button(
            "Update Password",
            use_container_width=True,
        )

    if submitted:

        if not current_password or not new_password or not confirm_password:

            st.error("Please fill in all password fields.")

        elif len(new_password) < 8:

            st.error(
                "Your new password must contain at least 8 characters."
            )

        elif new_password != confirm_password:

            st.error(
                "The new password and confirmation do not match."
            )

        elif current_password == new_password:

            st.warning(
                "Your new password must be different from your current password."
            )

        else:

            try:

                result = change_password(
                    username,
                    current_password,
                    new_password,
                )

                # Support either a tuple or a boolean return value.
                if isinstance(result, tuple):

                    success = bool(result[0])

                    message = (
                        str(result[1])
                        if len(result) > 1
                        else "Password update completed."
                    )

                else:

                    success = bool(result)

                    message = (
                        "Password updated successfully."
                        if success
                        else "Password update failed. Check your current password."
                    )

                if success:

                    try:
                        log_activity(
                            username,
                            "PASSWORD_CHANGED",
                            "User changed their account password.",
                        )
                    except Exception:
                        pass

                    st.success(message)

                    st.info(
                        "Your new password is now active. "
                        "Keep it private and do not share it with anyone."
                    )

                else:

                    st.error(message)

            except Exception as exc:

                st.error(
                    "The password could not be updated. "
                    "Please check the database.py function."
                )

                with st.expander("Technical details"):

                    st.code(str(exc))


# ============================================================
# END OF PART 10
# ============================================================

# ============================================================
# PART 11: ADMIN PANEL
# ============================================================

def render_admin_panel():
    """Display user management and activity logs for administrators."""

    if st.session_state.get("user_role") != "admin":
        st.error("Access denied. Only administrators can access this page.")
        return

    show_title(
        "🛡️ Admin Panel",
        "Manage registered accounts and review application activity.",
    )

    st.warning(
        "Account status changes affect the user's ability to log in. "
        "Review the selected account before making changes."
    )

    # --------------------------------------------------------
    # REFRESH
    # --------------------------------------------------------

    if st.button("🔄 Refresh Admin Data", key="admin_refresh"):

        st.rerun()

    # --------------------------------------------------------
    # REGISTERED USERS
    # --------------------------------------------------------

    st.subheader("👥 Registered Users")

    try:

        users = get_all_users()

        if users is None:

            users = []

        if isinstance(users, pd.DataFrame):

            users_df = users.copy()

        else:

            users_df = pd.DataFrame(users)

        if users_df.empty:

            st.info("No registered users were found.")

        else:

            # Never display password hashes.
            users_df = users_df.drop(
                columns=["password_hash", "password"],
                errors="ignore",
            )

            st.metric(
                "Total Registered Accounts",
                len(users_df),
            )

            if "role" in users_df.columns:

                admin_count = int(
                    (
                        users_df["role"].astype(str).str.lower()
                        == "admin"
                    ).sum()
                )

                user_count = int(
                    (
                        users_df["role"].astype(str).str.lower()
                        == "user"
                    ).sum()
                )

                metric1, metric2 = st.columns(2)

                with metric1:
                    st.metric("Administrators", admin_count)

                with metric2:
                    st.metric("Regular Users", user_count)

            show_dataframe(
                users_df,
                "admin_users_export.csv",
                height=350,
            )

            # ------------------------------------------------
            # ACTIVATE / DEACTIVATE USER
            # ------------------------------------------------

            st.divider()
            st.subheader("⚙️ Manage Account Status")

            required_columns = {"id", "username", "is_active"}

            if required_columns.issubset(set(users_df.columns)):

                manageable_users = users_df.copy()

                if "role" in manageable_users.columns:

                    # Do not offer administrators for deactivation.
                    manageable_users = manageable_users[
                        manageable_users["role"]
                        .astype(str)
                        .str.lower()
                        != "admin"
                    ]

                if manageable_users.empty:

                    st.info(
                        "No regular user accounts are available "
                        "for status changes."
                    )

                else:

                    user_options = {}

                    for _, row in manageable_users.iterrows():

                        label = (
                            f"{row['username']} "
                            f"(ID: {row['id']})"
                        )

                        user_options[label] = row.to_dict()

                    selected_label = st.selectbox(
                        "Select a user account",
                        options=list(user_options.keys()),
                        key="admin_selected_user",
                    )

                    selected_user = user_options[selected_label]

                    current_active = selected_user.get("is_active")

                    if isinstance(current_active, str):

                        current_active = (
                            current_active.strip().lower()
                            in {"1", "true", "yes", "active"}
                        )

                    else:

                        current_active = bool(current_active)

                    if current_active:

                        st.success("Selected account is currently active.")

                        desired_active = False

                        button_label = "Deactivate Selected Account"

                    else:

                        st.warning("Selected account is currently inactive.")

                        desired_active = True

                        button_label = "Activate Selected Account"

                    if st.button(
                        button_label,
                        key="admin_change_account_status",
                    ):

                        try:

                            result = set_user_active(
                                int(selected_user["id"]),
                                desired_active,
                            )

                            if isinstance(result, tuple):

                                status_success = bool(result[0])

                                status_message = (
                                    str(result[1])
                                    if len(result) > 1
                                    else "Account status updated."
                                )

                            else:

                                status_success = (
                                    result is not False
                                    and result is not None
                                )

                                status_message = (
                                    "Account status updated."
                                    if status_success
                                    else "Account status could not be updated."
                                )

                            if status_success:

                                try:

                                    log_activity(
                                        st.session_state["username"],
                                        "ADMIN_ACCOUNT_STATUS_CHANGED",
                                        (
                                            f"Changed account status for "
                                            f"{selected_user['username']} "
                                            f"to {'active' if desired_active else 'inactive'}."
                                        ),
                                    )

                                except Exception:

                                    pass

                                st.success(status_message)
                                st.rerun()

                            else:

                                st.error(status_message)

                        except Exception as exc:

                            st.error(
                                "Unable to update the selected account."
                            )

                            with st.expander("Technical details"):

                                st.code(str(exc))

            else:

                st.warning(
                    "Account status management requires the database "
                    "columns id, username and is_active."
                )

    except Exception as exc:

        st.error("Could not load registered users.")

        with st.expander("Technical details"):

            st.code(str(exc))

    # --------------------------------------------------------
    # ACTIVITY LOGS
    # --------------------------------------------------------

    st.divider()
    st.subheader("📋 Activity Logs")

    log_limit = st.selectbox(
        "Number of recent activity records",
        options=[25, 50, 100, 250, 500],
        index=2,
        key="admin_log_limit",
    )

    try:

        logs = get_activity_logs(limit=int(log_limit))

        if logs is None:

            logs = []

        if isinstance(logs, pd.DataFrame):

            logs_df = logs.copy()

        else:

            logs_df = pd.DataFrame(logs)

        if logs_df.empty:

            st.info("No activity records are available.")

        else:

            # Display newest records first when a timestamp is available.
            timestamp_column = next(
                (
                    column
                    for column in [
                        "created_at",
                        "timestamp",
                        "event_time",
                    ]
                    if column in logs_df.columns
                ),
                None,
            )

            if timestamp_column:

                logs_df = logs_df.sort_values(
                    timestamp_column,
                    ascending=False,
                )

            show_dataframe(
                logs_df,
                "admin_activity_logs.csv",
                height=400,
            )

    except Exception as exc:

        st.error("Could not load activity logs.")

        with st.expander("Technical details"):

            st.code(str(exc))

    # --------------------------------------------------------
    # SECURITY NOTICE
    # --------------------------------------------------------

    st.divider()

    st.caption(
        "Security: password hashes are not displayed in the user table. "
        "Administrators should use account-management features responsibly."
    )


# ============================================================
# END OF PART 11
# ============================================================

# ============================================================
# PART 12: MAIN APPLICATION ROUTING
# ============================================================

def main():
    """Connect authentication, navigation and application pages."""

    # --------------------------------------------------------
    # AUTHENTICATION
    # --------------------------------------------------------

    if not st.session_state.get("logged_in", False):
        login_screen()
        return

    # --------------------------------------------------------
    # SIDEBAR AND PAGE HEADER
    # --------------------------------------------------------

    selected_page = render_sidebar()

    render_page_header()

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    monthly, daily, p2p_p2m, apps = load_data_safely()

    # --------------------------------------------------------
    # PAGE ROUTING
    # --------------------------------------------------------

    if selected_page == "UPI Overview":

        render_overview(
            monthly,
            daily,
            p2p_p2m,
        )

    elif selected_page == "Daily & Monthly Analysis":

        render_daily_monthly_analysis(
            monthly,
            daily,
        )

    elif selected_page == "Application Analysis":

        render_application_analysis(
            apps,
        )

    elif selected_page == "Forecasting":

        render_forecasting(
            monthly,
        )

    elif selected_page == "Custom Data Analysis":

        render_custom_data_analysis()

    elif selected_page == "About":

        render_about()

    elif selected_page == "Change Password":

        render_change_password()

    elif selected_page == "Admin Panel":

        if st.session_state.get("user_role") == "admin":

            render_admin_panel()

        else:

            st.error(
                "Access denied. Only administrators can access this page."
            )

    else:

        st.warning(
            "The selected page is not available. "
            "Please choose a page from the sidebar."
        )


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":
    main()

# ============================================================
# END OF PART 12
# ============================================================

