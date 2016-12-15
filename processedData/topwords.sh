#!/bin/sh
sort -rn -k2 -t':' word_df.txt | head -$1 |  while read -r line;
do
  #echo "token id: $(echo $line | cut -d':' -f1) count : $(echo $line | cut -d':' -f2)" 
  #echo $line
  echo " \t $(grep ^$(echo $line | cut -d':' -f1)':' word_dict.txt | cut -d ':' -f2)"

done
  
