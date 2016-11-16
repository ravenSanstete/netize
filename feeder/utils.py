# this method can be put into utils
def migrate_db(clt_name,_set,name_list):
    conn=pymongo.MongoClient('localhost',27017);
    clt=conn.ml[clt_name];
    log_size=np.round(_set.shape[0]/20);
    print(clt);
    assert(_set.shape[1]==len(name_list));
    suc_count=0;
    for i in range(_set.shape[0]):
        single_doc=dict();
        for j in range(len(name_list)):
            single_doc[name_list[j]]=int(_set[i][j]);
        # to insert a single document into the specified document
        if(np.mod(i,log_size)):
            print("%f finished." % (i/_set.shape[0]));
        try:
            clt.insert_one(single_doc);
            suc_count+=1;
        except Exception as e:
            print(e);
            continue;
    print('Successfully migrate %d into collection %s' % (suc_count,clt_name));
    return;
