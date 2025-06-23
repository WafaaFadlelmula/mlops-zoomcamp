import datetime
import time
import random
import logging 
import uuid
import pytz
import pandas as pd
import io
import psycopg
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")

SEND_TIMEOUT = 10
rand = random.Random()

# Get database configuration from environment variables
DB_HOST = os.getenv('DB_HOST', 'localhost')
DB_PORT = os.getenv('DB_PORT', '5432')
DB_USER = os.getenv('DB_USER', 'postgres')
DB_PASSWORD = os.getenv('DB_PASSWORD', 'example')
DB_NAME = os.getenv('DB_NAME', 'test')

create_table_statement = """
drop table if exists dummy_metrics;
create table dummy_metrics(
	timestamp timestamp,
	value1 integer,
	value2 varchar,
	value3 float
)
"""

def prep_db():
	# Connect to PostgreSQL server (not specific database)
	conn_string = f"host={DB_HOST} port={DB_PORT} user={DB_USER} password={DB_PASSWORD}"
	
	with psycopg.connect(conn_string, autocommit=True) as conn:
		# Check if database exists, create if not
		res = conn.execute("SELECT 1 FROM pg_database WHERE datname=%s", (DB_NAME,))
		if len(res.fetchall()) == 0:
			conn.execute(f"create database {DB_NAME};")
	
	# Connect to specific database and create table
	conn_string_with_db = f"host={DB_HOST} port={DB_PORT} user={DB_USER} password={DB_PASSWORD} dbname={DB_NAME}"
	with psycopg.connect(conn_string_with_db, autocommit=True) as conn:
		conn.execute(create_table_statement)

def calculate_dummy_metrics_postgresql(curr):
	value1 = rand.randint(0, 1000)
	value2 = str(uuid.uuid4())
	value3 = rand.random()

	curr.execute(
		"insert into dummy_metrics(timestamp, value1, value2, value3) values (%s, %s, %s, %s)",
		(datetime.datetime.now(pytz.timezone('Europe/London')), value1, value2, value3)
	)

def main():
	logging.info("Starting dummy metrics calculation...")
	
	# Wait for database to be ready
	max_retries = 30
	for i in range(max_retries):
		try:
			prep_db()
			logging.info("Database prepared successfully")
			break
		except psycopg.OperationalError as e:
			if i < max_retries - 1:
				logging.info(f"Database not ready, retrying in 2 seconds... ({i+1}/{max_retries})")
				time.sleep(2)
			else:
				logging.error(f"Failed to connect to database after {max_retries} attempts: {e}")
				return
	
	last_send = datetime.datetime.now() - datetime.timedelta(seconds=10)
	conn_string = f"host={DB_HOST} port={DB_PORT} dbname={DB_NAME} user={DB_USER} password={DB_PASSWORD}"
	
	with psycopg.connect(conn_string, autocommit=True) as conn:
		for i in range(0, 100):
			with conn.cursor() as curr:
				calculate_dummy_metrics_postgresql(curr)

			new_send = datetime.datetime.now()
			seconds_elapsed = (new_send - last_send).total_seconds()
			if seconds_elapsed < SEND_TIMEOUT:
				time.sleep(SEND_TIMEOUT - seconds_elapsed)
			while last_send < new_send:
				last_send = last_send + datetime.timedelta(seconds=10)
			logging.info(f"data sent - iteration {i+1}")

if __name__ == '__main__':
	main()