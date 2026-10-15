
import sqlite3
import bcrypt
from pathlib import Path
from datetime import datetime

# Database file will be created in the project folder.
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "upi_system.db"


def get_connection():
    """Create a connection to the SQLite database."""
    connection = sqlite3.connect(DB_PATH, timeout=10)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def hash_password(password):
    """Securely hash a password using bcrypt."""
    return bcrypt.hashpw(
        password.encode("utf-8"),
        bcrypt.gensalt()
    ).decode("utf-8")


def verify_password(password, password_hash):
    """Verify a password against its stored hash."""
    try:
        return bcrypt.checkpw(
            password.encode("utf-8"),
            password_hash.encode("utf-8")
        )
    except (ValueError, TypeError, AttributeError):
        return False


def log_activity(username, event, details=""):
    """Record an activity without storing any password."""
    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO activity_log
                (username, event, details, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (
                username,
                event,
                details,
                datetime.now().isoformat(timespec="seconds")
            )
        )


def create_user(username, email, password, role="user"):
    """Create a new user account."""
    username = username.strip()
    email = email.strip().lower()

    if not username or not email or not password:
        return False, "Username, email and password are required."

    if len(username) < 3:
        return False, "Username must contain at least 3 characters."

    if len(password) < 8:
        return False, "Password must contain at least 8 characters."

    if role not in ("user", "admin"):
        return False, "Invalid user role."

    password_hash = hash_password(password)

    try:
        with get_connection() as connection:
            connection.execute(
                """
                INSERT INTO users
                    (username, email, password_hash, role,
                     is_active, created_at)
                VALUES (?, ?, ?, ?, 1, ?)
                """,
                (
                    username,
                    email,
                    password_hash,
                    role,
                    datetime.now().isoformat(timespec="seconds")
                )
            )

        log_activity(username, "ACCOUNT_CREATED", f"Role: {role}")
        return True, "Account created successfully."

    except sqlite3.IntegrityError:
        return False, "Username or email already exists."


def authenticate_user(username, password):
    """Authenticate an active user."""
    username = username.strip()

    with get_connection() as connection:
        user = connection.execute(
            """
            SELECT id, username, email, password_hash, role, is_active
            FROM users
            WHERE username = ?
            """,
            (username,)
        ).fetchone()

    if user is None or not user["is_active"]:
        log_activity(username, "LOGIN_FAILED", "Invalid or inactive account")
        return None

    if not verify_password(password, user["password_hash"]):
        log_activity(username, "LOGIN_FAILED", "Incorrect password")
        return None

    log_activity(username, "LOGIN_SUCCESS", "Login successful")

    return {
        "id": user["id"],
        "username": user["username"],
        "email": user["email"],
        "role": user["role"]
    }


def change_password(username, current_password, new_password):
    """Change a user's password after verifying the current one."""
    if len(new_password) < 8:
        return False, "New password must contain at least 8 characters."

    with get_connection() as connection:
        user = connection.execute(
            """
            SELECT password_hash FROM users
            WHERE username = ? AND is_active = 1
            """,
            (username.strip(),)
        ).fetchone()

        if user is None or not verify_password(
            current_password, user["password_hash"]
        ):
            return False, "Current password is incorrect."

        new_hash = hash_password(new_password)
        connection.execute(
            "UPDATE users SET password_hash = ? WHERE username = ?",
            (new_hash, username.strip())
        )

    log_activity(username, "PASSWORD_CHANGED", "Password updated")
    return True, "Password changed successfully."


def get_all_users():
    """Return user details without password hashes."""
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT id, username, email, role, is_active, created_at
            FROM users
            ORDER BY id
            """
        ).fetchall()

    return [dict(row) for row in rows]


def set_user_active(user_id, is_active):
    """Activate or deactivate a user account."""
    with get_connection() as connection:
        cursor = connection.execute(
            "UPDATE users SET is_active = ? WHERE id = ? AND role != 'admin'",
            (1 if is_active else 0, user_id)
        )

    return cursor.rowcount == 1


def get_activity_logs(limit=100):
    """Return recent activity records."""
    limit = max(1, min(int(limit), 1000))

    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT id, username, event, details, created_at
            FROM activity_log
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,)
        ).fetchall()

    return [dict(row) for row in rows]


def initialize_db():
    """Create tables and seed an initial admin if none exists."""
    with get_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                email TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'user'
                    CHECK(role IN ('user', 'admin')),
                is_active INTEGER NOT NULL DEFAULT 1
                    CHECK(is_active IN (0, 1)),
                created_at TEXT NOT NULL
            )
            """
        )

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS activity_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                event TEXT NOT NULL,
                details TEXT DEFAULT '',
                created_at TEXT NOT NULL
            )
            """
        )

        admin_exists = connection.execute(
            "SELECT 1 FROM users WHERE role = 'admin' LIMIT 1"
        ).fetchone()

        if admin_exists is None:
            connection.execute(
                """
                INSERT INTO users
                    (username, email, password_hash, role,
                     is_active, created_at)
                VALUES (?, ?, ?, 'admin', 1, ?)
                """,
                (
                    "Parul",
                    "admin@upi.local",
                    hash_password("Parul@123"),
                    datetime.now().isoformat(timespec="seconds")
                )
            )

            connection.execute(
                """
                INSERT INTO activity_log
                    (username, event, details, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (
                    "admin",
                    "ADMIN_CREATED",
                    "Initial administrator account created",
                    datetime.now().isoformat(timespec="seconds")
                )
            )


if __name__ == "__main__":
    initialize_db()
    print("Database initialized successfully.")
    print(f"Database location: {DB_PATH}")
    print("Users table and activity log are ready.")
    print("Change the temporary admin password before real use.")
