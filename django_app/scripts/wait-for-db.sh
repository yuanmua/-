#!/bin/bash
#
##host="$1"
##shift
##port="$1"
##shift
##
##echo "Waiting for $host:$port..."
##while ! nc -z $host $port; do
##  sleep 0.5
##done
##
##echo "$host:$port is available, executing command"
##exec "$@"
#
#
#
host="$1"
shift
port="$2"
shift


echo "Waiting for $host..."
while ! curl -s "$host" > /dev/null; do
  sleep 2
done

sleep 2
echo "$host:$port is available, executing command"
exec "$@"
