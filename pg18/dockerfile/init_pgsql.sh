#!/bin/bash

initdb -D /var/lib/pgsql/18.3/data --username="postgres" --pwfile="/var/lib/pgsql/18.3/passwd"

# No TCP/IP: listen on Unix socket only. Connect via: docker exec -it <ctr> psql -U postgres
echo "listen_addresses = ''" >> /var/lib/pgsql/18.3/data/postgresql.conf

sed -i 's/max_wal_size = 1GB/max_wal_size = 50GB/g' /var/lib/pgsql/18.3/data/postgresql.conf

# Load all benchmark-specific settings from the dedicated config file
echo "include = '/var/lib/pgsql/18.3/ceb_benchmark.conf'" >> /var/lib/pgsql/18.3/data/postgresql.conf

# Install pg_buffercache in template1 so all future databases inherit it
pg_ctl start -D /var/lib/pgsql/18.3/data -w -o "-c listen_addresses=''"
psql -d template1 -U postgres -c "CREATE EXTENSION pg_buffercache;"
pg_ctl stop -D /var/lib/pgsql/18.3/data -w -m fast
