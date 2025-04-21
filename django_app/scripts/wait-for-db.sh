##!/bin/bash
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
#host="$1"
#shift
#port="$2"
#shift
#
#
#echo "Waiting for $host:3306..."
#while ! curl -s "$host:$port" > /dev/null; do
#  sleep 2
#done
#
#echo "$host:$port is available, executing command"
#exec "$@"
