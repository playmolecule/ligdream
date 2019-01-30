echo "Retrieving the zip file with the model weights . . . "
wget http://pub.htmd.org/ligdream-20190128T143457Z-001.zip .

echo "Extracting files . . . "
unzip ligdream-20190128T143457Z-001.zip

if [ -d modelweights ]   # for file "if [-f /home/rama/file]" 
then 
    rm -r modelweights
fi
mv ligdream modelweights
rm ligdream-20190128T143457Z-001.zip

echo "Completed"
