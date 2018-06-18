FROM tensorflow/tensorflow:latest

RUN pip --no-cache-dir install \
 keras \
 tflearn \
 tensorflowjs \
 plotly

RUN mkdir -p /static

ADD scaffolding/ /static/
ADD entrypoint.sh /usr/local/bin

RUN chmod 0755 /usr/local/bin/entrypoint.sh

EXPOSE 8080

WORKDIR "/workdir"

CMD ["/usr/local/bin/entrypoint.sh"]