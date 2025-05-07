program patch

   use netcdf

implicit none

   character(len=80) :: filename
   integer, dimension(3) :: src_idx
   real*8, dimension(3) :: weights
   integer*1, dimension(1) :: valid
   integer :: ncid, rc, varid

   filename = 'mpas_to_healpix_weights_order9.nc'

   ! south pole, hp cell 2621441 at zoom 9
   valid = 1
   src_idx = (/13,655430,655429/) ! these are mpas cell indices
   weights = (/0.3333,0.4764,0.1902/)
   weights(1) = 1._8 - (weights(2) + weights(3))

   rc = nf90_open(trim(filename),nf90_write,ncid)
   print*,'open ',rc
   
   rc = nf90_inq_varid(ncid,'valid',varid)
   print*,'inq_varid valid ',rc
   rc = nf90_put_var(ncid,varid,valid,(/2621441/),(/1/))
   print*,'put valid ',rc

   rc = nf90_inq_varid(ncid,'src_idx',varid)
   print*,'inq_varid src_idx ',rc
   rc = nf90_put_var(ncid,varid,src_idx,(/1,2621441/),(/3,1/))
   print*,'put src_idx ',rc

   rc = nf90_inq_varid(ncid,'weights',varid)
   print*,'inq_varid weights ',rc
   rc = nf90_put_var(ncid,varid,weights,(/1,2621441/),(/3,1/))
   print*,'put weights ',rc

   ! north pole, hp cell 786432 at zoom 9
   !    - a mirror image of south pole so same weights used
   src_idx = (/25,655500,655501/)
   rc = nf90_inq_varid(ncid,'valid',varid)
   print*,'inq_varid valid ',rc
   rc = nf90_put_var(ncid,varid,valid,(/786432/),(/1/))
   print*,'put valid ',rc

   rc = nf90_inq_varid(ncid,'src_idx',varid)
   print*,'inq_varid src_idx ',rc
   rc = nf90_put_var(ncid,varid,src_idx,(/1,786432/),(/3,1/))
   print*,'put src_idx ',rc

   rc = nf90_inq_varid(ncid,'weights',varid)
   print*,'inq_varid weights ',rc
   rc = nf90_put_var(ncid,varid,weights,(/1,786432/),(/3,1/))
   print*,'put weights ',rc



   rc = nf90_close(ncid)
   print*,'close ',rc

end program patch
