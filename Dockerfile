FROM tensorflow/tensorflow:latest

# RUN apt-get update && apt-get install -y \
#    python-tk \
#    && rm -rf /var/lib/apt/lists/*

RUN pip --no-cache-dir install \
 keras \
 tflearn \
 tensorflowjs \
 plotly

RUN mkdir -p /workdir/static

ADD workdir/ /workdir/static/
ADD entrypoint.sh /usr/local/bin

RUN chmod 0755 /usr/local/bin/entrypoint.sh

EXPOSE 8080

WORKDIR "/workdir"

CMD ["/usr/local/bin/entrypoint.sh"]