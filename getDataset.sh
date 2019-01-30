echo "Retrieving cleaned ZINC15 dataset . . . "
wget http://pub.htmd.org/zinc15_druglike_clean_canonical_max60.zip

if ! [ -d traindataset ]
then 
   mkdir traindataset
fi

echo "Extracting files . . . "
unzip zinc15_druglike_clean_canonical_max60.zip -d traindataset

rm zinc15_druglike_clean_canonical_max60.zip
