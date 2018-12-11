# Counter for sample not in
rm eliminated.txt
counter=0;
for x in $(awk '{print $2}' freeze2_complete_GTE_Groningen_07092016.txt); do
    tmp=$(grep -c "${x}\$" bios_lld_rawdata_names.txt);
    if [ ${tmp} -ne 1 ]; then
        counter=$[ ${counter} + 1 ];
        grep "${x}\$" \
            freeze2_complete_GTE_Groningen_07092016.txt >> eliminated.txt;
    fi;
done;
echo ${counter}