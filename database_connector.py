"""sql database connector"""
import time
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.exc import ProgrammingError, SQLAlchemyError
from sqlalchemy.engine import URL
from logging_handler import Logger
import traceback

class SQLDataBase:
    """Class to connect to the SQL DataBase (Postgres)"""

    def __init__(self, config):
        try:
            self.config = config
            self.engine, self.conn = self.get_db_connection()
            Logger.info("Database connection initialised successfully")
        except Exception as e:
            Logger.error("Database connection initialisation failed")
            Logger.error(traceback.format_exc())


    def get_db_connection(self, ):
        """Get db connection"""
        url_object = URL.create(
            "postgresql+psycopg2",
            username=self.config["user"],
            password=self.config["password"],
            host=self.config["host"],
            port=self.config["port"],
            database=self.config["database"]
        )

        engine = create_engine(url_object, connect_args={"options": "-c search_path={}".format(self.config["schema"])})
        conn = engine.connect()
        Logger.info("connection with database engine built")

        return engine, conn
    
    @property
    def dialect(self):
        """Return string representation of dialect to use."""
        return self.conn.dialect.name
 
    def reconnect(self, retry_limit=5):
        counter = 0
        while counter < retry_limit:
            Logger.info("retry:{} for connection with database".format(retry_limit))
            _, self.conn = self.get_db_connection()
            if self.is_conn_closed() == False:
                break
            time.sleep(180) # retry after 3 mins
            counter += 1
    
    def is_conn_closed(self, ):
        if self.conn.closed == 1:
            Logger.info("Database connection is closed")
            return True
        else:
            Logger.info("Database connection is opened")
            return False

    def execute_sql(self, sql_query):
        """function to execute the SQL query"""
        try:
            data = pd.read_sql_query(
                sql_query,
                self.conn)
            self.conn.close()
            Logger.info("Database connection has been closed")
            return data
        except SQLAlchemyError as e:
            """Format the error message"""
            Logger.error(f"SQLchaemy Error: {e}")
            self.conn.close()
            Logger.info("Database connection has been closed")
            return f"Error: {e}"
        except Exception as e1:
            Logger.error("Pandas read_sql_query failed")
            self.conn.close()
            Logger.info("Database connection has been closed")
            Logger.error(traceback.format_exc())
            return f"Error: {e1}"

    
    def execute_sql_cursor(self, sql_query: str, fetch: str = "all") -> str:
        """Execute a SQL command and return a string representing the results.
        If the statement returns rows, a string of the results is returned.
        If the statement returns no rows, an empty string is returned.
        """
        with self.engine.begin() as connection:
            _schema = self.config["schema"]
            if _schema is not None:
                connection.exec_driver_sql(f"SET search_path TO {_schema}")
            cursor = connection.execute(text(sql_query))
            if cursor.returns_rows:
                if fetch == "all":
                    result = cursor.fetchall()
                elif fetch == "one":
                    result = cursor.fetchone()[0]
                else:
                    raise ValueError("Fetch parameter must be either 'one' or 'all'")
                return str(result)
        return ""