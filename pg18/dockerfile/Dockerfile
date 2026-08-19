FROM debian:bookworm

RUN apt-get update && apt-get install -y apt-transport-https ca-certificates

RUN apt-get update && apt-get install -y \
    wget \
    git \
    gcc \
    build-essential \
    libreadline-dev \
    zlib1g-dev \
    bison \
    flex \
    libicu-dev \
    libssl-dev \
    pkg-config

COPY ./postgresql-18.3.tar.gz /root/postgresql-18.3.tar.gz

RUN cd /root/ \
    && tar xvf postgresql-18.3.tar.gz \
    && rm postgresql-18.3.tar.gz \
    && cd postgresql-18.3 \
    && ./configure --prefix=/usr/local/pgsql/18.3 --enable-depend --enable-cassert --enable-debug CFLAGS="-ggdb -O0" \
    && make \
    && make install \
    && cd contrib/pg_buffercache \
    && make \
    && make install \
    && cd /root/postgresql-18.3 \
    && echo 'export PATH=/usr/local/pgsql/18.3/bin:$PATH' >> /root/.bashrc \
    && echo 'export LD_LIBRARY_PATH=/usr/local/pgsql/18.3/lib/:$LD_LIBRARY_PATH' >> /root/.bashrc

# pg_hint_plan: use ossc-db/pg_hint_plan PG18 branch (Nathaniel-Han fork was PG13-only)
RUN git clone --depth=1 --branch PG18 https://github.com/ossc-db/pg_hint_plan /root/pg_hint_plan \
    && cd /root/pg_hint_plan \
    && make PG_CONFIG=/usr/local/pgsql/18.3/bin/pg_config \
    && make PG_CONFIG=/usr/local/pgsql/18.3/bin/pg_config install

ENV PATH $PATH:/usr/local/pgsql/18.3/bin
ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:/usr/local/pgsql/18.3/lib/

RUN set -eux \
	&& groupadd -r postgres --gid=1001 \
	&& useradd -r -g postgres --uid=1001 --home-dir=/var/lib/pgsql/18.3/data --shell=/bin/bash postgres \
	&& mkdir -p /var/lib/pgsql/18.3/data \
	&& chown -R postgres:postgres /var/lib/pgsql/18.3/data \
	&& echo 'postgres' > /var/lib/pgsql/18.3/passwd \
	&& chmod -R 777 /var/lib/pgsql/18.3/passwd

COPY --chown=postgres:postgres ./ceb_benchmark.conf /var/lib/pgsql/18.3/ceb_benchmark.conf
COPY ./init_pgsql.sh /var/lib/pgsql/18.3/init_pgsql.sh
RUN chmod a+x /var/lib/pgsql/18.3/init_pgsql.sh

USER postgres
RUN /var/lib/pgsql/18.3/init_pgsql.sh

CMD ["postgres", "-D", "/var/lib/pgsql/18.3/data"]
