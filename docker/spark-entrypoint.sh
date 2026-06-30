#!/bin/bash
# CLASSPATH completo (Hadoop + Spark) calculado en runtime: depende del
# contenido final de ambos directorios de jars, no de lo que habia al
# construir la imagen. Sin esto, pyarrow.fs.FileSystem (usado por
# `cargar_imagen_hdfs`) no encuentra las clases Java que `libhdfs` invoca via
# JNI y falla con NoClassDefFoundError al primer acceso real a HDFS.
set -e

export CLASSPATH="$("${HADOOP_HOME}/bin/hadoop" classpath --glob):$(echo "${SPARK_HOME}/jars"/*.jar | tr ' ' ':')"

exec "$@"
