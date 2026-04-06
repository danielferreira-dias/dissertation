"""
Organize Kaggle DermNet images by actual condition using filename parsing.

DermNet folders group multiple conditions together (e.g., "Psoriasis pictures
Lichen Planus and related diseases"). This script parses filenames to extract
the real condition and organizes images into clean class folders.
"""

import re
import shutil
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "kaggle_dermnet"
OUTPUT_DIR = PROJECT_ROOT / "data" / "dataset" / "kaggle_dermnet"

# Filename prefix -> target class mapping
# Prefixes are matched case-insensitively against the start of filenames
# Order matters: longer/more specific prefixes first
FILENAME_TO_CLASS = [
    # Acne
    ("acne-vulgaris", "acne"),
    ("07Acne", "acne"),
    ("07acne", "acne"),
    ("acne-", "acne"),
    ("Comedones", "acne"),
    ("comedones", "acne"),
    # Rosacea
    ("07Rosacea", "rosacea"),
    ("07rosacea", "rosacea"),
    ("07Rhinophyma", "rosacea"),
    ("07rhnophyma", "rosacea"),
    ("rhinophyma", "rosacea"),
    ("rosacea", "rosacea"),
    # Perioral Dermatitis
    ("07Perioral", "perioral_dermatitis"),
    ("08Perioral", "perioral_dermatitis"),
    ("perioral-dermatitis", "perioral_dermatitis"),
    # Seborrheic Dermatitis
    ("08SebDerm", "seborrheic_dermatitis"),
    ("08SebDErm", "seborrheic_dermatitis"),
    ("07sebDer", "seborrheic_dermatitis"),
    ("seborrheic-dermatitis", "seborrheic_dermatitis"),
    # Psoriasis
    ("08Psoriasis", "psoriasis"),
    ("psoriasis", "psoriasis"),
    # Lichen Planus
    ("08LichenPlanus", "lichen_planus"),
    ("lichen-planus", "lichen_planus"),
    # LSA (Lichen Sclerosus)
    ("08LSA", "lichen_sclerosus"),
    # Basal Cell Carcinoma
    ("basal-cell-carcinoma", "basal_cell_carcinoma"),
    # Actinic Keratosis
    ("actinic-keratosis", "actinic_keratosis"),
    ("actinic-cheilitis", "actinic_keratosis"),
    ("arsenical-keratoses", "actinic_keratosis"),
    # Squamous Cell Carcinoma
    ("squamous-cell-carcinoma", "squamous_cell_carcinoma"),
    # Melanoma
    ("malignant-melanoma", "melanoma"),
    ("lentigo-maligna", "melanoma"),
    ("nail-melanoma", "melanoma"),
    # Atypical Nevi / Moles
    ("atypical-nevi", "atypical_nevi"),
    ("melanocytic-nevi", "melanocytic_nevi"),
    ("congenital-nevus", "congenital_nevus"),
    ("halo-nevus", "halo_nevus"),
    ("blue-nevus", "blue_nevus"),
    ("spitz-nevus", "spitz_nevus"),
    ("becker-nevus", "becker_nevus"),
    ("nevus-spilus", "nevus_spilus"),
    # Eczema
    ("eczema-", "eczema"),
    ("03Eczema", "eczema"),
    ("03eczema", "eczema"),
    ("03Dermatitis", "eczema"),
    ("03ContactDerm", "eczema"),
    ("03ChronicSta", "eczema"),
    ("03AnalExcor", "eczema"),
    ("03Desquamation", "eczema"),
    ("Dyshidrosis", "eczema"),
    ("desquamation", "eczema"),
    ("chapped-fissured", "eczema"),
    ("diaper-rash", "eczema"),
    # Atopic Dermatitis
    ("05Atopic", "atopic_dermatitis"),
    ("05ATopicAreola", "atopic_dermatitis"),
    ("05DennieMorgan", "atopic_dermatitis"),
    ("05DryCracked", "atopic_dermatitis"),
    ("05DryFeet", "atopic_dermatitis"),
    ("05hyperkeratosis", "atopic_dermatitis"),
    ("03Icthyosis", "atopic_dermatitis"),
    ("03ichthyosis", "atopic_dermatitis"),
    # Contact Dermatitis
    ("allergic-contact-dermatitis", "contact_dermatitis"),
    ("irritant-contact-dermatitis", "contact_dermatitis"),
    ("rhus-dermatitis", "contact_dermatitis"),
    ("cement-dermatitis", "contact_dermatitis"),
    ("shoe-allergy", "contact_dermatitis"),
    ("metal-dermatitis", "contact_dermatitis"),
    ("cosmetic-fragrance", "contact_dermatitis"),
    ("contact-airborne", "contact_dermatitis"),
    ("contact-dermatitis", "contact_dermatitis"),
    ("natural-rubber-latex", "contact_dermatitis"),
    ("patch-testing", "contact_dermatitis"),
    ("angry-back", "contact_dermatitis"),
    ("systemically-induced", "contact_dermatitis"),
    ("Poison-Ivy", "contact_dermatitis"),
    # Urticaria
    ("06Hives", "urticaria"),
    ("hives-Urticaria", "urticaria"),
    ("angioedema", "urticaria"),
    ("dermagraphism", "urticaria"),
    ("cholinergic-ur", "urticaria"),
    ("cold-urticaria", "urticaria"),
    ("pressure-urticaria", "urticaria"),
    # PUPPP (pregnancy-related, separate)
    ("PUPPP", "puppp"),
    # Urticaria Vasculitis
    ("Urticaria-Vasculitis", "urticaria_vasculitis"),
    # Scabies
    ("scabies", "scabies"),
    # Lyme Disease
    ("lyme-disease", "lyme_disease"),
    ("acrodermatitis-chronica", "lyme_disease"),
    # Insect Bites
    ("biting-insects", "insect_bite"),
    ("flea-bites", "insect_bite"),
    ("fire-ants", "insect_bite"),
    ("ants", "insect_bite"),
    ("chigger-bites", "insect_bite"),
    ("cat-bite", "insect_bite"),
    ("cat-scratch", "insect_bite"),
    ("jelly-fish", "insect_bite"),
    ("15JellyFish", "insect_bite"),
    ("15SeaBather", "insect_bite"),
    ("Sea-bathers", "insect_bite"),
    ("Coral-Poison", "insect_bite"),
    ("caterpillar", "insect_bite"),
    ("duck-itch", "insect_bite"),
    ("cutaneous-Larva", "insect_bite"),
    ("tick-bite", "insect_bite"),
    # Tinea / Fungal
    ("13Tinea", "tinea"),
    ("tinea-", "tinea"),
    # Candida
    ("13Candida", "candida"),
    ("13candida", "candida"),
    ("Candida", "candida"),
    # Intertrigo
    ("13Intertrigo", "intertrigo"),
    ("13Intertirgo", "intertrigo"),
    ("13intertrigo", "intertrigo"),
    ("13intertrig", "intertrigo"),
    # Folliculitis
    ("Folliculitis", "folliculitis"),
    ("folliculitis", "folliculitis"),
    ("09Pseudomonas", "folliculitis"),
    ("Pseudomonas-Folliculitis", "folliculitis"),
    ("Staphylococcal-Folliculitis", "folliculitis"),
    ("Sycosis-Barbae", "folliculitis"),
    # Cellulitis
    ("Cellulitis", "cellulitis"),
    ("09cellulitis", "cellulitis"),
    ("Pseudomonas-Cellulitis", "cellulitis"),
    ("cellulitis", "cellulitis"),
    # Impetigo
    ("Ecthyma", "impetigo"),
    ("Erysipelas", "impetigo"),
    ("impetigo", "impetigo"),
    ("Staphylococal-Diaper", "impetigo"),
    ("Staphylococcal-Scalded", "impetigo"),
    # Hidradenitis
    ("Hidradenitis", "hidradenitis"),
    ("hidradenitis", "hidradenitis"),
    # Herpes Simplex
    ("herpes-simplex", "herpes_simplex"),
    ("herpes-type-1", "herpes_simplex"),
    ("herpes-type-2", "herpes_simplex"),
    ("herpes-cutaneous", "herpes_simplex"),
    ("herpes-immunocomp", "herpes_simplex"),
    ("herpes-Buttock", "herpes_simplex"),
    ("herpes-buttocks", "herpes_simplex"),
    ("herpetic-Whitlow", "herpes_simplex"),
    ("Herpes-Hand", "herpes_simplex"),
    ("eczema-herpeticum", "herpes_simplex"),
    ("genital-herpes", "herpes_simplex"),
    ("11herpesAnal", "herpes_simplex"),
    # Herpes Zoster
    ("herpes-zoster", "herpes_zoster"),
    # Warts
    ("12Wart", "warts"),
    ("12wart", "warts"),
    ("10Warts", "warts"),
    ("warts-", "warts"),
    ("genital-warts", "warts"),
    ("11AnalWarts", "warts"),
    ("bowenoid-papulosis", "warts"),
    # Molluscum
    ("molluscum", "molluscum"),
    # Vitiligo
    ("vitiligo", "vitiligo"),
    # Melasma
    ("melasma", "melasma"),
    # Post-inflammatory hyperpigmentation
    ("post-inflammatory-hyperpigmentation", "pih"),
    # Seborrheic Keratosis
    ("seborrheic-keratosis", "seborrheic_keratosis"),
    ("seborrheic-keratoses-smooth", "seborrheic_keratosis"),
    ("seborrheic-keratoses-ruff", "seborrheic_keratosis"),
    ("stucco-keratoses", "seborrheic_keratosis"),
    ("dermatosis-papulosa-nigra", "seborrheic_keratosis"),
    # Dermatofibroma
    ("dermatofibroma", "dermatofibroma"),
    # Epidermal Cyst
    ("epidermal-cyst", "epidermal_cyst"),
    ("20CystClear", "epidermal_cyst"),
    ("20EpidermCyst", "epidermal_cyst"),
    ("20PilarCyst", "epidermal_cyst"),
    ("20RupturedCyst", "epidermal_cyst"),
    ("20cystAnal", "epidermal_cyst"),
    # Keloid
    ("keloid", "keloid"),
    ("acne-keloidalis", "keloid"),
    # Drug Eruption
    ("drug-eruption", "drug_eruption"),
    ("drug-lichenoid", "drug_eruption"),
    ("fixed-drug-eruption", "drug_eruption"),
    ("minocycline-pigment", "drug_eruption"),
    # Erythema Multiforme
    ("erythema-multiforme", "erythema_multiforme"),
    # Stevens-Johnson
    ("stevens-johnson", "stevens_johnson"),
    ("toxic-epidermal", "stevens_johnson"),
    # Lupus
    ("lupus-", "lupus"),
    ("discoid-lupus", "lupus"),
    # Kaposi Sarcoma
    ("kaposi-sarcoma", "kaposi_sarcoma"),
    # Pyogenic Granuloma
    ("pyogenic-granuloma", "pyogenic_granuloma"),
    ("granuloma-pyogenic", "pyogenic_granuloma"),
    # Granuloma Annulare
    ("granuloma-annulare", "granuloma_annulare"),
    # Pityriasis Rosea
    ("pityriasis-rosea", "pityriasis_rosea"),
    # Keratosis Pilaris
    ("keratosis-pilaris", "keratosis_pilaris"),
    # Acanthosis Nigricans
    ("acanthosis-nigricans", "acanthosis_nigricans"),
    # Alopecia
    ("alopecia-areata", "alopecia_areata"),
    ("androgenic-alopecia", "androgenic_alopecia"),
    ("traction-alopecia", "traction_alopecia"),
    # Hemangioma
    ("hemangioma", "hemangioma"),
    # Vasculitis
    ("vasculitis", "vasculitis"),
    # Prurigo Nodularis
    ("prurigo-nodularis", "prurigo_nodularis"),
    # Stasis Dermatitis
    ("stasis-", "stasis_dermatitis"),
    # Bullous Pemphigoid
    ("bullous-pemphigoid", "bullous_pemphigoid"),
    # Pemphigus
    ("pemphigus", "pemphigus"),
    # Darier's Disease
    ("dariers-disease", "dariers_disease"),
    ("darier-", "dariers_disease"),
    # Epidermolysis Bullosa
    ("epidermolysis-bullosa", "epidermolysis_bullosa"),
    # Scleroderma / Morphea
    ("scleroderma", "scleroderma"),
    ("morphea", "scleroderma"),
    # Dermatomyositis
    ("dermatomyositis", "dermatomyositis"),
    # Erythema Nodosum
    ("erythema-nodosum", "erythema_nodosum"),
    # Livedo Reticularis
    ("livido-reticularis", "livedo_reticularis"),
    ("livedo-reticularis", "livedo_reticularis"),
    # Onychomycosis / Nail Fungus
    ("distal-subungual-onycho", "onychomycosis"),
    ("onychomycosis", "onychomycosis"),
    # Varicella / Chickenpox
    ("12Chicken", "varicella"),
    ("12varicella", "varicella"),
    ("varicella", "varicella"),
    # Monkey Pox
    ("monkey-pox", "monkeypox"),
    # Small Pox
    ("small-pox", "smallpox"),
    # Psoriasis (additional patterns from unclassified)
    ("Psoriasis-Guttate", "psoriasis"),
    ("Psoriasis-Chronic-Plaque", "psoriasis"),
    ("Psoriasis-Hand", "psoriasis"),
    ("Psoriasis-inversus", "psoriasis"),
    ("Psoriasis-nails", "psoriasis"),
    ("Psoriasis-penis", "psoriasis"),
    ("Psoriasis-Anus", "psoriasis"),
    ("Psoriasis-Histology", "psoriasis"),
    # Lichen Planus (additional)
    ("Lichen-Planus-Oral", "lichen_planus"),
    ("Lichen-Planus-Drug", "lichen_planus"),
    ("Lichen-Planus-Hypertrophic", "lichen_planus"),
    ("Lichen-Planus-Penis", "lichen_planus"),
    # Lichen Sclerosus (additional)
    ("Lichen-Sclerosus", "lichen_sclerosus"),
    ("Lichen-Sclerosis", "lichen_sclerosus"),
    # Lichen Simplex Chronicus
    ("lichen-simplex-chronicus", "lichen_simplex"),
    # Lichen Nitidus
    ("Lichen-Nitidus", "lichen_nitidus"),
    # Keratoacanthoma
    ("keratoacanthoma", "keratoacanthoma"),
    # Bowen's Disease
    ("bowens-disease", "bowens_disease"),
    # Sun Damaged Skin
    ("sun-damaged-skin", "sun_damage"),
    # Sebaceous Hyperplasia
    ("sebaceous-hyperplasia", "sebaceous_hyperplasia"),
    ("SebaceousAdemoma", "sebaceous_hyperplasia"),
    ("Sebaceoushyperplasia", "sebaceous_hyperplasia"),
    # CTCL (Cutaneous T-cell Lymphoma)
    ("ctcl", "ctcl"),
    # Mucous Cyst
    ("mucous-cyst", "mucous_cyst"),
    # Viral Exanthems
    ("viral-exanthems", "viral_exanthem"),
    ("enterovirus", "viral_exanthem"),
    ("scarlet-fever", "viral_exanthem"),
    ("erythema-infectiosum", "viral_exanthem"),
    ("gianotti-crosti", "viral_exanthem"),
    ("kawasaki-syndrome", "viral_exanthem"),
    ("hand-foot-mouth", "viral_exanthem"),
    ("unilateral-laterothoracic", "viral_exanthem"),
    ("roseola", "viral_exanthem"),
    ("measles", "viral_exanthem"),
    # Phototoxic Reactions
    ("phototoxic-reactions", "phototoxic"),
    ("polymorphous-light-eruption", "phototoxic"),
    ("erythema-ab-igne", "phototoxic"),
    ("actinic-comedones", "phototoxic"),
    # Skin Tags
    ("skin-tags-polyps", "skin_tags"),
    # Dermatitis Herpetiformis
    ("dermatitis-herpetiformis", "dermatitis_herpetiformis"),
    # Necrobiosis Lipoidica
    ("necrobiosis-lipoidica", "necrobiosis_lipoidica"),
    # Schamberg Disease
    ("schamberg-disease", "schamberg_disease"),
    # Chondrodermatitis
    ("chondrodermatitis-nodularis", "chondrodermatitis"),
    # Porokeratosis
    ("porokeratosis", "porokeratosis"),
    # Xanthomas
    ("xanthomas", "xanthomas"),
    # Epidermal Nevus
    ("epidermal-nevus", "epidermal_nevus"),
    ("nevus-sebaceous", "epidermal_nevus"),
    # Neurofibromatosis
    ("neurofibromatosis", "neurofibromatosis"),
    ("neurofibromas", "neurofibromatosis"),
    # Pityriasis Rubra Pilaris
    ("pityriasis-rubra-pilaris", "pityriasis_rubra_pilaris"),
    # Pityriasis Lichenoides
    ("Pityriasis-Lichenoides", "pityriasis_lichenoides"),
    # Perleche / Angular Cheilitis
    ("perleche", "perleche"),
    ("13Perleche", "perleche"),
    ("13perleche", "perleche"),
    ("03AngularCheilitis", "perleche"),
    # Onycholysis
    ("onycholysis", "onycholysis"),
    # Neurotic Excoriations
    ("neurotic-excoriations", "neurotic_excoriations"),
    ("biting-excoriation", "neurotic_excoriations"),
    # Tuberous Sclerosis
    ("tuberous-sclerosis", "tuberous_sclerosis"),
    # Grovers Disease
    ("grovers-disease", "grovers_disease"),
    # Candida (additional patterns)
    ("candidiasis-large-skin-folds", "candida"),
    ("candidiasis-diaper", "candida"),
    ("candidiasis-mouth", "candida"),
    ("candida-penis", "candida"),
    ("candida-groin", "candida"),
    ("erosio-interdigitalis", "candida"),
    # Pilar Cyst
    ("pilar-cyst", "epidermal_cyst"),
    # Keratolysis Exfoliativa
    ("keratolysis-exfoliativa", "keratolysis_exfoliativa"),
    # Intertrigo (additional)
    ("intertrigo", "intertrigo"),
    # ID Reaction
    ("id-reaction", "id_reaction"),
    # Syringoma
    ("syringoma", "syringoma"),
    # AIDS
    ("AIDS", "aids_related"),
    # Porphyrias
    ("porphyrias", "porphyria"),
    # Venous
    ("venous-malformations", "venous_malformation"),
    ("venous-lake", "venous_lake"),
    # Angiokeratomas
    ("angiokeratomas", "angiokeratoma"),
    # Spider Bite
    ("spider-bite", "insect_bite"),
    # Spider Angioma
    ("spider-angioma", "spider_angioma"),
    # Cherry Angioma
    ("cherry-angioma", "cherry_angioma"),
    # Telangiectasias
    ("telangiectasias", "telangiectasia"),
    # Accessory Nipple
    ("accessory-nipple", "accessory_nipple"),
    ("accessory-trachus", "accessory_nipple"),
    # Henoch-Schonlein Purpura
    ("henoch-schonlein", "henoch_schonlein"),
    # Pyoderma Gangrenosum
    ("pyoderma-gangrenosum", "pyoderma_gangrenosum"),
    ("Pyoderma-Gangrenosum", "pyoderma_gangrenosum"),
    ("Atypical-Pyoderma", "pyoderma_gangrenosum"),
    # Panniculitis
    ("panniculitis", "panniculitis"),
    # Majocchi Purpura
    ("majocchi-purpura", "majocchi_purpura"),
    # Furuncles
    ("Furuncles-Carbuncles", "furuncle"),
    # Benign Familial Chronic Pemphigus (Hailey-Hailey)
    ("benign-familial-chronic-pemphigus", "hailey_hailey"),
    # Atopic Dermatitis (additional missed patterns)
    ("05atopic", "atopic_dermatitis"),
    ("05atopicFeet", "atopic_dermatitis"),
    ("05keratosisPilaris", "keratosis_pilaris"),
    ("05KeratosisPilaris", "keratosis_pilaris"),
    # Steroid Rosacea
    ("07Steroid", "rosacea"),
    # Vascular Face
    ("07Vascular", "rosacea"),
    ("23Vessels", "telangiectasia"),
    # Milia
    ("milia", "milia"),
    # Fordyce Spots
    ("fordyce-spots", "fordyce_spots"),
    ("Sebaceous-glands", "fordyce_spots"),
    # Melanoma Mimic
    ("melanoma-mimic", "melanoma_mimic"),
    # Basal Cell Nevus Syndrome
    ("basal-cell-nevus-syndrome", "basal_cell_carcinoma"),
    # Cutaneous Horn
    ("cutaneous-horn", "cutaneous_horn"),
    # Diabetic Bullae
    ("diabetic-bullae", "diabetic_bullae"),
    ("diabetes-mellitus", "diabetic_skin"),
    # Nail conditions
    ("beaus-lines", "nail_disease"),
    ("acute-paronychia", "paronychia"),
    ("chronic-paronychia", "paronychia"),
    ("yellow-nails", "nail_disease"),
    ("pincer-nails", "nail_disease"),
    ("nail-distal-splitting", "nail_disease"),
    ("ingrown-nail", "nail_disease"),
    ("twenty-nail-dystrophy", "nail_disease"),
    ("median-nail-dystrophy", "nail_disease"),
    ("ridging-beading", "nail_disease"),
    ("splinter-hemorrhage", "nail_disease"),
    ("pigmented-bands", "nail_disease"),
    ("onychogryphosis", "nail_disease"),
    ("koilonychia", "nail_disease"),
    ("clubbing", "nail_disease"),
    ("color-changes", "nail_disease"),
    ("congenital-anomalies", "nail_disease"),
    ("dry-nails", "nail_disease"),
    ("fissure", "nail_disease"),
    ("habit-tic-deformity", "nail_disease"),
    ("hang-nail", "nail_disease"),
    ("leukonychia", "nail_disease"),
    ("normal-variations", "nail_disease"),
    ("white-superficial-onychomycosis", "onychomycosis"),
    ("proximal-subungual-onychomycosis", "onychomycosis"),
    ("eczema-nail", "eczema"),
    # Infected Eczema (bacterial)
    ("09EczemaInfected", "eczema"),
    ("09EczemaStaph", "eczema"),
    # Herpes Gestationis
    ("herpes-gestationis", "herpes_gestationis"),
    # Leprosy
    ("Leprosy", "leprosy"),
    ("Lupus-vulgaris", "leprosy"),
    # Syphilis
    ("syphilis", "syphilis"),
    # Gonorrhea
    ("gonorrhea", "gonorrhea"),
    # Reiter Syndrome
    ("reiter-syndrome", "reiter_syndrome"),
    # Pompholyx
    ("pompholyx", "eczema"),
    # Sarcoid
    ("sarcoid", "sarcoidosis"),
    # Pretibial Myxedema
    ("pretibial-myxedema", "pretibial_myxedema"),
    # Amyloidosis
    ("amyloidosis", "amyloidosis"),
    # Erythema Annulare Centrifugum
    ("erythema-annulare-centrifugum", "erythema_annulare"),
    # Crest Syndrome
    ("crest-syndrome", "scleroderma"),
    # Periungual Warts
    ("periungual-warts", "warts"),
    # Paget Disease
    ("paget-disease", "paget_disease"),
    # Majocchi Granuloma
    ("majocchi-granuloma", "tinea"),
    # Granulation Tissue
    ("granulation-tissue", "granulation_tissue"),
    # Trauma
    ("trauma", "trauma"),
    # Kerion
    ("kerion", "tinea"),
    # Erythrasma
    ("erythrasma", "erythrasma"),
    # Pitted Keratolysis
    ("pitted-keratolysis", "pitted_keratolysis"),
    # Corns
    ("corns", "corns"),
    ("black-heel", "corns"),
    # Otitis Externa
    ("otitis-externa", "otitis_externa"),
    # Tufted Folliculitis
    ("tufted-folliculitis", "folliculitis"),
    # Dissecting Cellulitis
    ("dissecting-cellulitis", "folliculitis"),
    # Gram Negative Folliculitis
    ("gram-negative-folliculitis", "folliculitis"),
    # Lentigo
    ("lentigo-adults", "lentigo"),
    # Idiopathic Guttate Hypomelanosis
    ("idiopathic-guttate-hypomelanosis", "idiopathic_guttate_hypomelanosis"),
    # Metastasis
    ("metastasis", "cutaneous_metastasis"),
    # Lipoid Proteinosis
    ("lipoid-proteinosis", "lipoid_proteinosis"),
    # Various IMG prefixed (numbered DermNet images)
    ("11IMG", "misc_dermnet"),
    ("9IMG", "misc_dermnet"),
    ("8IMG", "misc_dermnet"),
    ("7IMG", "misc_dermnet"),
    ("6IMG", "misc_dermnet"),
    ("5T", "misc_dermnet"),
    ("1IMG", "misc_dermnet"),
    # Remaining niche
    ("Erysipeloid", "impetigo"),
    ("Botryomycosis", "impetigo"),
    ("Streptococci-Anal", "impetigo"),
    ("9sporotrichoid", "atypical_mycobacteria"),
    ("HotTub", "folliculitis"),
    ("atypical-mycobacterium", "atypical_mycobacteria"),
    ("chilblains-perniosis", "chilblains"),
    ("Rocky-mountain", "rocky_mountain_spotted_fever"),
    ("rocky-mountain", "rocky_mountain_spotted_fever"),
    ("maculae-cerulea", "pediculosis"),
    ("pubic-lice", "pediculosis"),
    ("pediculosis-", "pediculosis"),
    ("head-lice", "pediculosis"),
    ("myiasis", "myiasis"),
    ("leishmaniasis", "leishmaniasis"),
    ("dermatitis-swimming", "swimmers_itch"),
    ("leukoplakia", "leukoplakia"),
    ("pearly-penile-papules", "pearly_penile_papules"),
    ("nevus-anemicus", "nevus_anemicus"),
    ("hyperhidrosis", "hyperhidrosis"),
    ("anal-Comedones", "acne"),
    ("nevus-comedonicus", "nevus_comedonicus"),
    ("07PerlecheAccutane", "perleche"),
    ("Forest", "misc_dermnet"),
    ("condyloma", "warts"),
    ("bacterial-vaginosis", "bacterial_vaginosis"),
    ("genital-ulcers", "genital_ulcers"),
    ("granuloma-inguinale", "granuloma_inguinale"),
    ("lymphogranuloma", "lymphogranuloma"),
    ("chancroid", "chancroid"),
    # Alopecia (additional)
    ("folliculitis-decalvans", "folliculitis"),
    ("lichen-planopilaris", "lichen_planus"),
    ("pseudopelade", "alopecia_other"),
    ("hot-comb-alopecia", "traction_alopecia"),
    ("telogen-effluvium", "alopecia_other"),
    ("anagen-effluvium", "alopecia_other"),
    ("hirsutism", "hirsutism"),
    ("monilethrix", "hair_disease"),
    ("pili-annulati", "hair_disease"),
    ("polytrichia", "hair_disease"),
    ("sheathed-hair", "hair_disease"),
    ("trichomycosis", "hair_disease"),
    ("trichorrhexis", "hair_disease"),
    ("ingrown-eyelash", "hair_disease"),
    # Connective Tissue (misc)
    ("acrocyanosis", "acrocyanosis"),
    ("atrophoderma", "atrophoderma"),
    ("dupuytren-contracture", "dupuytren"),
    ("erythromelalgia", "erythromelalgia"),
    ("macular-atrophy", "macular_atrophy"),
    ("mixed-connective-tissue", "mctd"),
    ("raynaud-disease", "raynaud"),
    ("reflex-sympathetic", "crps"),
    ("temporal-arteritis", "temporal_arteritis"),
    ("rheumatoid-nodule", "rheumatoid_nodule"),
    # Systemic (misc)
    ("03dermatitisDrug", "drug_eruption"),
    ("26BirtHogg", "birt_hogg_dube"),
    ("26Fibrofolliculoma", "birt_hogg_dube"),
    ("26birtHog", "birt_hogg_dube"),
    ("Eosinophilic-Granuloma", "histiocytosis"),
    ("Eruptive-Xanthoma", "xanthomas"),
    ("Hans-Schuller", "histiocytosis"),
    ("Leser-Trelat", "leser_trelat"),
    ("Letterer-Siwe", "histiocytosis"),
    ("letterer-siwe", "histiocytosis"),
    ("Relapsing-Polychondritis", "polychondritis"),
    ("addison-disease", "addison"),
    ("bechet", "behcet"),
    ("cowden-disease", "cowden"),
    ("cryoglobulinemia", "cryoglobulinemia"),
    ("degos-disease", "degos"),
    ("elastosis-perferans", "elastosis"),
    ("glucagonoma", "glucagonoma"),
    ("gout", "gout"),
    ("histiocytosis", "histiocytosis"),
    ("hypothyroidism", "hypothyroidism"),
    ("klinefelter", "klinefelter"),
    ("lichen-myxedematosus", "scleromyxedema"),
    # Vascular Tumors (misc)
    ("HomaniomacIMG", "hemangioma"),
    ("cutis-marmorata", "cutis_marmorata"),
    ("av-malformation", "av_malformation"),
    ("thrombosed-vein", "thrombosed_vein"),
    ("unilateral-telangiectasia", "telangiectasia"),
    ("vascular-anomaly", "vascular_anomaly"),
    ("purpura-vomiting", "purpura"),
    # Interstitial Granulomatous Dermatitis
    ("interstitial-granulomatous", "granulomatous_dermatitis"),
    # Polyarteritis Nodosa
    ("polyarteritis-nodosa", "polyarteritis_nodosa"),
    # Atrophy Blanche
    ("atrophy-blanche", "atrophy_blanche"),
    # Granuloma Faciale
    ("granuloma-faciale", "granuloma_faciale"),
    # Sweets Syndrome
    ("sweets-syndrome", "sweets_syndrome"),
    # Eccrine / Adnexal
    ("eccrine-spiradenoma", "eccrine_spiradenoma"),
    ("cylindroma", "cylindroma"),
    ("calcifying-epithelioma", "pilomatricoma"),
    ("pilomatricoma", "pilomatricoma"),
    ("connective-tissue-nevus", "connective_tissue_nevus"),
    ("acquired-digital-fibrokeratoma", "fibrokeratoma"),
    ("atypical-fibroxanthoma", "fibroxanthoma"),
    ("endometriosis", "endometriosis"),
    ("extraneous-digits", "supernumerary_digit"),
    ("fibroma", "fibroma"),
    ("fibromatosis", "fibromatosis"),
    ("follicular-mucinosis", "follicular_mucinosis"),
    ("granular-cell-tumor", "granular_cell_tumor"),
    # Pseudomonas (misc)
    ("pseudomonas", "pseudomonas"),
    ("Balanitis-Bacterial", "balanitis"),
    # Contact dermatitis (additional in other folder)
    ("contact-dermatitis-leg", "contact_dermatitis"),
    # Nail specific
    ("Darier-nails", "dariers_disease"),
    ("lichen-planus", "lichen_planus"),
    # Sebaceous Nevus
    ("20SebNevus", "epidermal_nevus"),
    # Giant Comedone
    ("07GiantComedone", "acne"),
    # Hidradenitis in acne folder
    ("Hidradenitis-Suppurativa", "hidradenitis"),
]


def classify_filename(filename: str) -> str | None:
    """Match a filename to a condition class using prefix patterns."""
    for prefix, cls in FILENAME_TO_CLASS:
        if filename.startswith(prefix):
            return cls
    return None


def organize():
    """Scan all DermNet images and organize by actual condition."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    classified = Counter()
    unclassified = Counter()
    total = 0

    for split in ["train", "test"]:
        split_dir = RAW_DIR / split
        if not split_dir.exists():
            print(f"Warning: {split_dir} not found")
            continue

        for folder in sorted(split_dir.iterdir()):
            if not folder.is_dir():
                continue

            for img in folder.iterdir():
                if not img.is_file() or img.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
                    continue

                total += 1
                cls = classify_filename(img.name)

                if cls:
                    dest = OUTPUT_DIR / cls / img.name
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    if not dest.exists():
                        shutil.copy2(img, dest)
                    classified[cls] += 1
                else:
                    unclassified[f"{folder.name}/{img.name}"] = 1

    # Print summary
    print("\n" + "=" * 60)
    print("KAGGLE DERMNET — ORGANIZED BY CONDITION")
    print("=" * 60)

    grand_total = 0
    for cls, count in sorted(classified.items(), key=lambda x: -x[1]):
        print(f"  {cls:40s} {count:>5d} images")
        grand_total += count

    print(f"\n  {'CLASSIFIED TOTAL':40s} {grand_total:>5d} images")
    print(f"  {'UNCLASSIFIED':40s} {len(unclassified):>5d} images")
    print(f"  {'TOTAL SCANNED':40s} {total:>5d} images")
    print(f"\n  Output: {OUTPUT_DIR}")

    # Write unclassified to file for review
    if unclassified:
        unclass_file = OUTPUT_DIR / "_unclassified.txt"
        with open(unclass_file, "w") as f:
            for name in sorted(unclassified.keys()):
                f.write(name + "\n")
        print(f"  Unclassified list: {unclass_file}")

    # Write README
    readme = OUTPUT_DIR / "README.md"
    with open(readme, "w") as f:
        f.write("# Kaggle DermNet Dataset\n\n")
        f.write("- **Source:** Kaggle (shubhamgoel27/dermnet)\n")
        f.write("- **Type:** Clinical photos from DermNet NZ\n")
        f.write("- **Labels:** Extracted from filenames (not folder names)\n")
        f.write("- **License:** CC BY-NC-ND 4.0\n")
        f.write(f"- **Total classified:** {grand_total}\n")
        f.write(f"- **Unclassified:** {len(unclassified)}\n\n")
        f.write("## Classes\n\n")
        f.write("| Class | Images |\n")
        f.write("|-------|--------|\n")
        for cls, count in sorted(classified.items(), key=lambda x: -x[1]):
            f.write(f"| {cls} | {count:,} |\n")

    print(f"  README: {readme}")


if __name__ == "__main__":
    organize()
