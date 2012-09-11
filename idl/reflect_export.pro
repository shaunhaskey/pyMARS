shot='148765'
shots=['146398', '146397', '146392'];,'148765']
fn='/u/haskeysr/hdf5testfile10.h5'
;fid = H5F_OPEN(fn)  ;; file id
fid = H5F_CREATE(fn)
;fid = H5D_READ(fn)
;s = H5_PARSE(fn, /READ_DATA)
;HELP, s.146392_DATA, /STRUCTURE

FOR i=0, 2 DO BEGIN
    print,shots[i]
    tmp_data = get_refl(LONG(shots[i]))
    help, /str, tmp_data.profs
    a = {dens:tmp_data.profs.dens, time:tmp_data.profs.time, r:tmp_data.profs.r, rho:tmp_data.profs.rho}
    datatypeID = H5T_IDL_CREATE(a)
    dataspaceID = H5S_CREATE_SIMPLE(1)
    datasetID = H5D_CREATE(fid, shots[i], datatypeID, dataspaceID)
    H5D_WRITE, datasetID, a
ENDFOR

H5F_CLOSE, fid

end
