i=8
end=20
name=PC_long_maps_30min
while [ $i -le $end ]; do
    python3 runstrat.py \
    --strategy AI \
    --model-path /home/ane4ka/PySymGym/model.onnx \
    --timeout 1800 \
    -ps /home/ane4ka/PySymGym \
    --assembly-infos /home/ane4ka/PySymGym/maps/DotNet/Maps/Root/bin/Release/net7.0 /home/ane4ka/PySymGym/tools/runstrat/prebuilt/david.csv \
    --savedir /home/ane4ka/PySymGym/results/$name/$i
    i=$(($i+1))
done