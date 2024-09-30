select * from TB_MINSSTOCK tm ;
select * from TB_MINSSTOCK tm where tm.GSTC_CODE = 'KR7005930003' and tm.STCK_BSOP_DATE = '20240819';

SELECT MAX(a.STCK_CNTG_HOUR) AS STCK_CNTG_HOUR
FROM TB_MINSSTOCK a
WHERE a.GSTC_CODE = 'KR7005930003'
order by a.STCK_BSOP_DATE DESC ;

SELECT a.GSTC_CODE, a.STCK_BSOP_DATE , a.STCK_CNTG_HOUR , a.INSERTDATE 
FROM TB_MINSSTOCK a
where a.STCK_BSOP_DATE = '20240903'
order by a.INSERTDATE DESC ;

select *
  from TB_KRNEWS tk 
 order by KRNEWS_DATE DESC 
 limit 20;
 
SELECT *
  FROM TB_STOCKCLASSIFY ts ;
  
select *
  from TB_DAILYSTOCK td 
<<<<<<< HEAD
 order by STCK_BSOP_DATE ASC ;
=======
 where td.STCK_BSOP_DATE = '20240823';
>>>>>>> 257d464d1a7f4c3a6f1334ef70f0d9a5fb4f854f

SELECT GSTC_CODE , count(*)
  from TB_MINSSTOCK tm 
 where STCK_BSOP_DATE = '20240904'
   and STCK_CNTG_HOUR = '130000'
 group by GSTC_CODE 
 having count(*)>1;

select count(*)
  from TB_MINSSTOCK tm 
<<<<<<< HEAD
 where STCK_BSOP_DATE = '20240919'
   and STCK_CNTG_HOUR = '150000';
=======
 where STCK_BSOP_DATE = '20240909';
--    and STCK_CNTG_HOUR = '150000';
>>>>>>> 257d464d1a7f4c3a6f1334ef70f0d9a5fb4f854f
 
select a.GSTC_CODE, a.STCK_PRPR, a.CNTG_VOL
  from TB_MINSSTOCK a
 where a.GSTC_CODE = 'KR7042700005'
   and a.stck_cntg_hour >= '090000'
   and a.stck_cntg_hour <= '110000'
 order by a.stck_bsop_date asc, a.stck_cntg_hour asc;
 
select a.GSTC_CODE, a.STCK_PRPR, a.CNTG_VOL, a.STCK_BSOP_DATE, a.STCK_CNTG_HOUR 
  from TB_MINSSTOCK a
 where a.GSTC_CODE = 'KR7000660001'
   and ((a.stck_cntg_hour >= '140000' && a.STCK_BSOP_DATE = '2024082') || (a.stck_cntg_hour <= '110000' && a.STCK_BSOP_DATE = '20240829')) 
 order by a.stck_bsop_date asc, a.stck_cntg_hour asc;
 
select a.GSTC_CODE, a.STCK_PRPR, a.CNTG_VOL, a.STCK_BSOP_DATE, a.STCK_CNTG_HOUR 
  from TB_MINSSTOCK a
 where a.GSTC_CODE = 'KR7000660001'
   and a.stck_cntg_hour >= '140000'
   and a.STCK_BSOP_DATE = '20240802'
UNION all
select a.GSTC_CODE, a.STCK_PRPR, a.CNTG_VOL, a.STCK_BSOP_DATE, a.STCK_CNTG_HOUR 
  from TB_MINSSTOCK a
 where a.GSTC_CODE = 'KR7000660001'
   and a.stck_cntg_hour <= '110000'
   and a.STCK_BSOP_DATE = '20240805'
   
 order by stck_bsop_date asc, stck_cntg_hour asc
;