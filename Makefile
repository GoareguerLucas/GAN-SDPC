default: fetch_frioul

tmp_database:
	mkdir -p /tmp/database/ && rsync -a "/Users/laurentperrinet/science/VB_These/Rapport d'avancement/database/Face_DataBase/Raw_DataBase" /tmp/database/

fetch_delete_frioul:
	rsync -av -u --delete --exclude .AppleDouble --exclude .git perrinet.l@frioul.int.univ-amu.fr:/hpc/invibe/perrinet.l/ICLR/HULK/cache_dir .

fetch_frioul:
	rsync -av -u --exclude .AppleDouble --exclude .git perrinet.l@frioul.int.univ-amu.fr:/hpc/invibe/perrinet.l/ICLR/HULK/cache_dir .
