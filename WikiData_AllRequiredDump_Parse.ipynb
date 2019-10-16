#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import codecs
import csv
import bz2
import time
import json


# ### Defining Variables

# In[2]:


dump_path = r"C:\versioned\crd_cds\Analyses\2018 WikiGate\Project\Data\WikiData Dump"
save_path = r"C:\versioned\crd_cds\Analyses\2018 WikiGate\Project\Data\WikiData Data Dump as CSV"
_encode_ = "utf-8"


# Here we will read JSON file using python.
# ##### Link
# https://dumps.wikimedia.org/wikidatawiki/entities/latest-all.json.bz2
# ##### Requirement
# latest-all.json.bz2

# ### Defining Functions

# In[3]:


# Nicely formatted time string
def hms_string(sec_elapsed):
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60
    return "{}:{:>02}:{:>05.2f}".format(h, m, s)


def ent_values(ent):
    wd_type = ent["type"]
    wd_item = ent["id"]
    
    if ent["labels"].get("en", "not found") == "not found":
        wd_label = ""
    else:
        wd_label = ent["labels"]["en"]["value"]
    
    if ent["descriptions"].get("en", "not found") == "not found":
        wd_desc = ""
    else:
        wd_desc = ent["descriptions"]["en"]["value"]
    
    if ent["sitelinks"].get("enwiki", "not found") == "not found":
        wd_title = ""
    else:
        wd_title = ent["sitelinks"]["enwiki"]["title"]
        
    return([wd_type, wd_item, wd_label, wd_desc, wd_title])


def concat_claims(claims):
    for rel_id, rel_claims in claims.items():
        for claim in rel_claims:
            yield claim


# ### Parsing

# In[4]:


latest_all_json = "latest-all.json.bz2"
filename = os.path.join(dump_path,latest_all_json)

WD_identification = "WD_identification_item.csv"
file_identification = os.path.join(save_path,WD_identification)

WD_wikibase_entityid = "WD_wikibase_entityid.csv"
file_wikibase_entityid  = os.path.join(save_path,WD_wikibase_entityid)

WD_quantity = "WD_quantity.csv"
file_quantity = os.path.join(save_path,WD_quantity)

WD_globecoordinate="WD_globecoordinate.csv"
file_globecoordinate = os.path.join(save_path,WD_globecoordinate)

WD_time="WD_time.csv"
file_time = os.path.join(save_path,WD_time)

i = 0

start_time = time.time()

with codecs.open(file_identification, "w", "utf-8") as op_identification,codecs.open(file_wikibase_entityid, "w", "utf-8") as op_wikibase_entityid,codecs.open(file_quantity, "w", "utf-8") as op_quantity,codecs.open(file_globecoordinate, "w", "utf-8") as op_globecoordinate,codecs.open(file_time, "w", "utf-8") as op_time:
    
    opw_identification = csv.writer(op_identification, quoting=csv.QUOTE_MINIMAL)
    opw_identification.writerow(["WD_Type", "WD_WikiData_Item",
                       "WD_Label", "WD_Description", "WD_Title"])
    
    opw_wikibase_entityid = csv.writer(op_wikibase_entityid, quoting=csv.QUOTE_MINIMAL)
    opw_wikibase_entityid.writerow(["WD_Subject","WD_Predicate","WD_Object"])    
    
    opw_quantity = csv.writer(op_quantity, quoting=csv.QUOTE_MINIMAL)
    opw_quantity.writerow(["WD_Subject","WD_Predicate","WD_Object","WD_Units"])    
    
    opw_globecoordinate = csv.writer(op_globecoordinate, quoting=csv.QUOTE_MINIMAL)
    opw_globecoordinate.writerow(["WD_Subject","WD_Predicate","WD_Object","WD_Precision"])
    
    opw_time = csv.writer(op_time, quoting=csv.QUOTE_MINIMAL)
    opw_time.writerow(["WD_Subject","WD_Predicate","WD_Object","WD_Precision"])
    
    
    with bz2.BZ2File(filename, "rb") as f:
        for line in f:
            line = line.decode('utf_8',errors="ignore")
            if i>1000000000:
                break
            elif line in ("[\n", "]\n"):
                pass
            else:
                ent = json.loads(line.rstrip('\n,'))
                
                if ent["type"] != "item":
                    continue
                    
                opw_identification.writerow(ent_values(ent))
                
                claims = concat_claims(ent["claims"])
                e1 = ent["id"]
            
                for claim in claims:
                    mainsnak = claim["mainsnak"]
                    rel = mainsnak["property"]
                    snak_datatype = mainsnak["datatype"]
                    
                    if mainsnak['snaktype'] == "value":
                        snak_value = mainsnak["datavalue"]["value"]
                        
                        if snak_datatype in ("wikibase-item", "wikibase-property"):
                            opw_wikibase_entityid.writerow([e1, rel, snak_value["id"]])
                            
                        elif snak_datatype == "quantity":
                            e2 = (snak_value["amount"],snak_value["unit"].strip(r"http://www.wikidata.org/entity/"))
                            opw_quantity.writerow([e1, rel, e2[0],e2[1]])
                            
                        elif snak_datatype == "globe-coordinate":
                            e2 = ((snak_value["latitude"],snak_value["longitude"]),snak_value["precision"])
                            opw_globecoordinate.writerow([e1, rel, e2[0], e2[1]])
                            
                        elif snak_datatype == "time":
                            e2 = (snak_value["time"],snak_value["precision"])
                            opw_time.writerow([e1, rel, e2[0],e2[1]])
                            
                        else:
                            pass            
                   
                
            if (i % 100000) == 0:
                    print("Total item processed: {:,}".format(i))
            i = i + 1

elapsed_time = time.time() - start_time
print("Elapsed time: {}".format(hms_string(elapsed_time)))

