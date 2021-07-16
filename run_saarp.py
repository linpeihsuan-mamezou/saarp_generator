from src.util import get_aia_filename_list, get_sharp_filename_list, find_close_sharpmap
from src.util import   Saarp_generator
from sunpy.map import Map
from aiapy.calibrate import register, update_pointing 



aia_dir_path = './data/AIAdata/*.fits'
sharp_dir_path = './data/SHARPdata/*.Br.fits'
aia_filename_list = get_aia_filename_list(aia_dir_path)
sharp_filename_list = get_sharp_filename_list(sharp_dir_path)



# a list corresponding to aia_filename_list, with the sharpmap at the close time
sharp_filename_list_use_to_rotate = [None for x in range(len(aia_filename_list))]

for aia_filename, i in zip(aia_filename_list, range(len(aia_filename_list))) : 
    sharp_filename_list_use_to_rotate[i] = find_close_sharpmap(aia_filename, sharp_filename_list)
    


# for aia_filename, sharp_filename in zip(aia_filename_list, sharp_filename_list_use_to_rotate): 
#     saarp = SaarpTransformer(aia_filename, sharp_filename)
#     saarp = saarp.make_saarp()


# calibrate AIA obs
# https://aiapy.readthedocs.io/en/latest/generated/gallery/prepping_level_1_data.html#sphx-glr-generated-gallery-prepping-level-1-data-py
aiamap = Map(aia_filename_list[0]) 
aiamap = register(update_pointing(aiamap)) 
sharpmap = Map(sharp_filename_list_use_to_rotate[0])

# generate saarp map
out_dir='saarp_npz/'
saarp = Saarp_generator(aiamap,sharpmap )
saarp_data = saarp.make_map()

# save the file 
saarp_data = saarp.save_saarp('Saarp_frame1.npz', out_dir)