default: fetch_frioul

tmp_database:
	mkdir -p /tmp/database/ && rsync -a "/Users/laurentperrinet/science/VB_These/Rapport d'avancement/database/Face_DataBase/Raw_DataBase" /tmp/database/

fetch_delete_frioul:
	rsync -av -u --delete --exclude .AppleDouble --exclude .git perrinet.l@frioul.int.univ-amu.fr:/hpc/invibe/perrinet.l/ICLR/HULK/cache_dir .

fetch_frioul:
	rsync -av -u --exclude .AppleDouble --exclude .git perrinet.l@frioul.int.univ-amu.fr:/hpc/invibe/perrinet.l/ICLR/HULK/cache_dir .

images:
	rsync

start_ssh:
	sudo systemctl start ssh

stop_ssh:
	sudo systemctl stop ssh

connect_ssh
	ssh albert@147.94.234.179

rsync_image:
	rsync -av aae/ albert@147.94.234.179:~/Bureau/Lucas/GAN-SDPC/ --exclude="models"