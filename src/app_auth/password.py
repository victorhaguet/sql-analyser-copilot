"""Password hashing utilities using argon2."""

from passlib.context import CryptContext # type: ignore

_pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")


def hash_password(password: str) -> str:
    """Encode the password

    Args:
        password (str): plain password

    Returns:
        str: Encoded password
    """
    return _pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password

    Args:
        plain_password (str): plain password
        hashed_password (str): encoded password

    Returns:
        bool: Confirm the encoded password correspond to the plain password
    """
    return _pwd_context.verify(plain_password, hashed_password)