program wrapmesh

   use netcdf

implicit none

   character(len=80) :: filein, fileout, attname
   integer :: ncells, ncells2
   real*8, dimension(:), allocatable :: ll, ll2
   integer :: rc, ncidin, ncidout, dimid, varid, natts, varid2
   integer :: n

   filein = 'hist_postproc_qu/EW_B2000_CAM7_15km_58L_2D_constant.nc'
   fileout = 'hist_postproc_qu/mesh_wrap.nc'

   rc = nf90_open(trim(filein),nf90_nowrite,ncidin)
   rc = nf90_create(trim(fileout),nf90_clobber,ncidout)
print*,'create',rc
   rc = nf90_inq_dimid(ncidin,'nCells',dimid)
   rc = nf90_inquire_dimension(ncidin,dimid,len=ncells)
   ncells2 = ncells*2
   allocate(ll(ncells))
   allocate(ll2(ncells2))

   rc = nf90_def_dim(ncidout,'nCells',ncells2,dimid)
   rc = nf90_def_var(ncidout,'lonCell',nf90_double,(/dimid/),varid2)
   rc = nf90_inq_varid(ncidin,'lonCell',varid)
   rc = nf90_inquire_variable(ncidin,varid,natts=natts)
   do n = 1,natts
      rc = nf90_inq_attname(ncidin,varid,n,attname)
      rc = nf90_copy_att(ncidin,varid,trim(attname),ncidout,varid2)
   enddo
   rc = nf90_def_var(ncidout,'latCell',nf90_double,(/dimid/),varid2)
   rc = nf90_inq_varid(ncidin,'latCell',varid)
   rc = nf90_inquire_variable(ncidin,varid,natts=natts)
   do n = 1,natts
      rc = nf90_inq_attname(ncidin,varid,n,attname)
      rc = nf90_copy_att(ncidin,varid,trim(attname),ncidout,varid2)
   enddo

   rc = nf90_enddef(ncidout)

   rc = nf90_inq_varid(ncidin,'lonCell',varid)
   rc = nf90_get_var(ncidin,varid,ll)
   ll2(1:ncells) = ll(1:ncells)
   ll2(ncells+1:ncells2) = ll(1:ncells) + 360._8
   rc = nf90_inq_varid(ncidout,'lonCell',varid)
   rc = nf90_put_var(ncidout,varid,ll2)

   rc = nf90_inq_varid(ncidin,'latCell',varid)
   rc = nf90_get_var(ncidin,varid,ll)
   ll2(1:ncells) = ll(1:ncells)
   ll2(ncells+1:ncells2) = ll(1:ncells)
   rc = nf90_inq_varid(ncidout,'latCell',varid)
   rc = nf90_put_var(ncidout,varid,ll2)

   rc = nf90_close(ncidout)

end program wrapmesh
