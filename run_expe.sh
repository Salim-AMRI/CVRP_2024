#!/bin/bash

for I in  X-n106-k14.vrp X-n110-k13.vrp X-n115-k10.vrp X-n120-k6.vrp X-n125-k30.vrp X-n129-k18.vrp X-n134-k13.vrp X-n139-k10.vrp X-n143-k7.vrp X-n148-k46.vrp X-n153-k22.vrp X-n157-k13.vrp X-n162-k11.vrp X-n167-k10.vrp X-n172-k51.vrp X-n176-k26.vrp X-n181-k23.vrp X-n186-k15.vrp X-n190-k8.vrp X-n195-k51.vrp X-n200-k36.vrp; do
for X in OX AOX EAX LOX GPX PR; do

sbatch runscript_array_job_CVRP.sh $I $X

done
done


