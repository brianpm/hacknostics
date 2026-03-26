program adjust_cweights
   use netcdf
implicit none

   character(len=80) :: filename
   integer :: ncid, varid, dimid, dimlen, rc, n, tgt_idx, m, l
!   integer*1, dimension(:), allocatable :: v
   integer, dimension(:,:), allocatable :: src_idx
   integer, parameter :: ncells = 2621442

   filename = 'mpas_to_healpix_weights_order9.nc'

   rc = nf90_open(trim(filename),nf90_write,ncid)

   rc = nf90_inq_dimid(ncid,'tgt_idx',dimid)
   rc = nf90_inquire_dimension(ncid,dimid,len=tgt_idx)
   print*,'size = ',tgt_idx

!   allocate(v(tgt_idx))
   allocate(src_idx(3,tgt_idx))

!   rc = nf90_inq_varid(ncid,'valid',varid)
!   rc = nf90_get_var(ncid,varid,v)

   rc = nf90_inq_varid(ncid,'src_idx',varid)
print*,'inqvarid ',rc
rc = nf90_enddef(ncid)
   rc = nf90_get_var(ncid,varid,src_idx)
print*,'get ',rc

   m = 0
   do n = 1,tgt_idx
!      if ( v(n) .ne. 1_1) then
!         m = m+1
!      endif
      do l = 1,3
         if(src_idx(l,n) >= ncells) src_idx(l,n) = src_idx(l,n) - ncells
      enddo
   enddo

   print*,'number of valid == 0)',m

   rc = nf90_put_var(ncid,varid,src_idx)

   rc = nf90_close(ncid)

end program adjust_cweights
