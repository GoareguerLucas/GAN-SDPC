default: rsync

clean_models:
	rm */models/*.pt; rm */*/models/*.pt

ssh_start:
	sudo systemctl start ssh

ssh_stop:
	sudo systemctl stop ssh

ssh_connect:
	ssh albert@147.94.234.202

connect_gt2:
	ssh -p8012 -L16007:localhost:6007 g14006889@gt-2.luminy.univ-amu.fr

connect_gt0:
	ssh -p8012 -L16006:localhost:6006 g14006889@gt-0.luminy.univ-amu.fr

matplot:
	export MPLCONFIGDIR="/var/lib/vz/data/g14006889/cache/matplotlib/"

scp_gt0:
	scp -P 8012 -r g14006889@gt-0.luminy.univ-amu.fr:/var/lib//vz/data/g14006889/GAN-SDPC/ .

weigth:
	du -h . --max-depth=1

count_file:
	ls -Al | wc -l

venv:
	source ../p3/bin/activate

pep8:
	autopep8 $(DIR)/*.py -r -i --max-line-length 120 --ignore E402

see_rsync:
	rsync -avhuzn . albert@139.124.208.130:~/Bureau/Lucas/GAN-SDPC/ --exclude-from=ExclusionRSync

rsync:
	rsync -avhuz . albert@139.124.208.130:~/Bureau/Lucas/GAN-SDPC/ --exclude-from=ExclusionRSync

meso_rsync:
	rsync --progress -avhuz --exclude-from=ExclusionRSync lperrinet@login.mesocentre.univ-amu.fr:/scratch/lperrinet/SDPC/GAN-SDPC/ .

babbage_rsync:
	rsync --progress -avhuz --exclude-from=ExclusionRSync laurent@10.164.7.21:science/GAN-SDPC/GAN-SDPC/ .
