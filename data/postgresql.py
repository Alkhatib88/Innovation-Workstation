import psycopg2
from contextlib import contextmanager

class PostgreSQL:
    def __init__(self, app):
        self.app = app
        self.database_name = None
        self.user = None
        self.password = None
        self.host = None
        self.port = None
        self.connection = None

    def setup(self):
        settings_arg_definitions = {
            'database_name': {'type': str, 'help': 'Name of the PostgreSQL database'},
            'user': {'type': str, 'help': 'Database user'},
            'password': {'type': str, 'help': 'Password for the database user'},
            'host': {'type': str, 'help': 'Host where the database is located'},
            'port': {'type': str, 'help': 'Port of the database server', 'default': '5432'}
        }
        self.app.command.add_command(command_name="sql", function=None, description="Main command for SQL operations", category="Database Systems")
        self.app.command.add_subcommand(main_command="sql", subcommand_name="connect", function=self.connect, description="Connects to PostgreSQL database", help="Use this command to establish a connection to the PostgreSQL database.")
        self.app.command.add_subcommand(main_command="sql", subcommand_name="settings", function=self.settings, description="Collect PostgreSQL server information", arg_definitions=settings_arg_definitions, help="Set the configuration for the PostgreSQL database connection.")

        # New subcommands for managing tables and databases
        self.app.command.add_subcommand(main_command="sql", subcommand_name="create_table", function=self.create_table, description="Create a new table", help="Create a new table in the database.")
        self.app.command.add_subcommand(main_command="sql", subcommand_name="drop_table", function=self.drop_table, description="Drop a table", help="Drop an existing table from the database.")
        self.app.command.add_subcommand(main_command="sql", subcommand_name="alter_table_add_column", function=self.alter_table_add_column, description="Alter table to add a column", help="Add a new column to an existing table.")
        self.app.command.add_subcommand(main_command="sql", subcommand_name="create_database", function=self.create_database, description="Create a new database", help="Create a new database.")
        self.app.command.add_subcommand(main_command="sql", subcommand_name="drop_database", function=self.drop_database, description="Drop a database", help="Drop an existing database.")
        self.app.command.add_subcommand(main_command="sql", subcommand_name="select", function=self.select, description="Perform a SELECT query", help="Perform a SELECT query on the database.")

    def settings(self, database_name, user, password, host, port='5432'):
        self.database_name = database_name
        self.user = user
        self.password = password
        self.host = host
        self.port = port
        self.app.logger.info("SQL settings updated")
        return "SQL settings updated"

    def connect(self):
        try:
            self.connection = psycopg2.connect(
                database=self.database_name,
                user=self.user,
                password=self.password,
                host=self.host,
                port=self.port
            )
            self.app.logger.info("Connected to SQL: Success")
            return "Connected to SQL: Success"
        except psycopg2.OperationalError as e:
            self.app.logger.error(f"Error connecting to the database: {e}")
            return f"Error connecting to the database: {e}"

    def get_connection(self):
        return self.connection

    def close_connection(self):
        if self.connection:
            self.connection.close()

    @contextmanager
    def cursor(self):
        if self.connection is None:
            self.connect()
        cur = self.connection.cursor()
        try:
            yield cur
        except Exception as e:
            self.connection.rollback()
            self.app.logger.error(f"Database operation failed: {e}")
            raise e
        else:
            self.connection.commit()
        finally:
            cur.close()

    # Additional functionalities for managing tables and databases
    def create_table(self, table_name, columns):
        with self.cursor() as cursor:
            cursor.execute(f"CREATE TABLE {table_name} ({columns})")

    def drop_table(self, table_name):
        with self.cursor() as cursor:
            cursor.execute(f"DROP TABLE IF EXISTS {table_name}")

    def alter_table_add_column(self, table_name, column_definition):
        with self.cursor() as cursor:
            cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_definition}")

    def create_database(self, db_name):
        with self.cursor() as cursor:
            cursor.execute(f"CREATE DATABASE {db_name}")

    def drop_database(self, db_name):
        with self.cursor() as cursor:
            cursor.execute(f"DROP DATABASE IF EXISTS {db_name}")

    # Enhanced SELECT Queries
    def select(self, table, columns="*", condition=None):
        query = f"SELECT {columns} FROM {table}"
        if condition:
            query += f" WHERE {condition}"
        with self.cursor() as cursor:
            cursor.execute(query)
            return cursor.fetchall()

    def select_with_join(self, tables, join_type, on_condition, columns="*", where_condition=None):
        query = f"SELECT {columns} FROM {tables} {join_type} ON {on_condition}"
        if where_condition:
            query += f" WHERE {where_condition}"
        with self.cursor() as cursor:
            cursor.execute(query)
            return cursor.fetchall()

    def select_with_aggregation(self, table, aggregate_function, group_by=None):
        query = f"SELECT {aggregate_function} FROM {table}"
        if group_by:
            query += f" GROUP BY {group_by}"
        with self.cursor() as cursor:
            cursor.execute(query)
            return cursor.fetchall()

# Usage
# app_instance = YourAppInstance()  # Replace with your actual app instance
# db = PostgreSQL(app_instance)
# db.setup('database_name', 'user', 'password', 'host')
# with db.cursor() as cursor:
#     cursor.execute("SELECT * FROM table_name")
