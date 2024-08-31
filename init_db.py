import psycopg2
from psycopg2 import sql
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

# Connection configuration
config = {
    'dbname': 'postgres',
    'user': 'postgres',
    'password': 'example',
    'host': 'localhost',
    'port': '5432'
}

new_dbname = 'rag-1234'

def create_database():
    # Connect to the default PostgreSQL database
    conn = psycopg2.connect(**config)
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cursor = conn.cursor()
    
    try:
        # Check if the database already exists
        cursor.execute(sql.SQL("SELECT 1 FROM pg_database WHERE datname = %s"), (new_dbname,))
        exists = cursor.fetchone()
        
        if not exists:
            # Create the new database if it doesn't exist
            cursor.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(new_dbname)))
            print(f"Database {new_dbname} successfully created")
        else:
            print(f"Database {new_dbname} already exists")
    finally:
        cursor.close()
        conn.close()

def setup_vector_extension():
    # Connect to the newly created database
    new_config = config.copy()
    new_config['dbname'] = new_dbname
    conn = psycopg2.connect(**new_config)
    cursor = conn.cursor()
    
    try:
        # Activate the vector extension if it doesn't exist
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
        
        # Create a sample table with a vector column
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS items (
                id SERIAL PRIMARY KEY,
                embedding vector(3)
            )
        """)
        conn.commit()
        print(f"Vector extension successfully activated and sample table created in database {new_dbname}")
    except psycopg2.Error as e:
        print(f"An error occurred: {e}")
        conn.rollback()
    finally:
        cursor.close()
        conn.close()

if __name__ == "__main__":
    create_database()
    setup_vector_extension()