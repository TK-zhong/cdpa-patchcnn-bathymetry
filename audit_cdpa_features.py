"""Audit saved CDPA grids against raw gravity products; no decomposition parameters inferred."""
from pathlib import Path
import numpy as np
import pandas as pd
from CDPA_split import RAW_REGIONS, COMPONENTS, resolve_raw_path
ROOT=Path(__file__).resolve().parent
def long_fraction(array, lat, lon, wavelength=50.0):
    yy,xx=np.meshgrid(np.linspace(-1,1,len(lat)),np.linspace(-1,1,len(lon)),indexing='ij')
    design=np.column_stack((np.ones(array.size),xx.ravel(),yy.ravel()))
    flat=array.ravel();valid=np.isfinite(flat)
    coef=np.linalg.lstsq(design[valid],flat[valid],rcond=None)[0]
    residual=np.where(valid,flat-design@coef,0).reshape(array.shape)
    tapered=residual*np.hanning(len(lat))[:,None]*np.hanning(len(lon))[None,:]
    power=abs(np.fft.fft2(tapered))**2
    dy=abs(np.diff(lat).mean())*111.195
    dx=abs(np.diff(lon).mean())*111.195*np.cos(np.deg2rad(np.mean(lat)))
    fy,fx=np.meshgrid(np.fft.fftfreq(len(lat),dy),np.fft.fftfreq(len(lon),dx),indexing='ij')
    frequency=np.hypot(fx,fy)
    power[0,0]=0
    return float(power[(frequency>0)&(frequency<=1/wavelength)].sum()/power.sum())

rows=[]
for region in RAW_REGIONS:
    west,east,south,north=map(float,region.split('_'))
    def read(path):
        f=pd.read_csv(path,index_col=0)
        lat=f.index.to_numpy(dtype=float);lon=f.columns.to_numpy(dtype=float)
        r=(lat>=south)&(lat<=north);c=(lon>=west)&(lon<=east)
        return f.to_numpy(dtype=float)[r][:,c],lat[r],lon[c],f.shape
    for component in COMPONENTS:
        c1,lat,lon,original_shape=read(ROOT/'data'/region/f'standard_{component}_common.csv')
        c2,_,_,_=read(ROOT/'data'/region/f'swot_{component}_common.csv')
        d1,_,_,_=read(ROOT/'data'/region/f'standard_{component}_distinct.csv')
        d2,_,_,_=read(ROOT/'data'/region/f'swot_{component}_distinct.csv')
        x1,la,lo,_=read(resolve_raw_path(ROOT/'data','sio',component,region))
        x2,lb,lob,_=read(resolve_raw_path(ROOT/'data','swot',component,region))
        assert np.allclose(lat,la) and np.allclose(lat,lb) and np.allclose(lon,lo) and np.allclose(lon,lob)
        mask=np.logical_and.reduce([np.isfinite(a) for a in (x1,x2,c1,c2,d1,d2)])
        a,b,u,v,e,f=[a[mask] for a in (x1,x2,c1,c2,d1,d2)]
        alpha=float(u@v/(u@u))
        row=dict(region=region,component=component,stored_grid=str(original_shape),cropped_grid=str(c1.shape),n_grid=int(mask.sum()),
                 raw_corr=float(np.corrcoef(a,b)[0,1]),common_corr=float(np.corrcoef(u,v)[0,1]),
                 common_relative_difference=float(np.linalg.norm(u-v)/np.linalg.norm(u)),
                 common_scale_swot_over_sio=alpha,
                 common_scaled_residual=float(np.linalg.norm(v-alpha*u)/np.linalg.norm(v)),
                 common_standardized_difference=float(np.sqrt(np.mean(((u-u.mean())/u.std()-(v-v.mean())/v.std())**2))),
                 sio_reconstruction_relative=float(np.linalg.norm(a-u-e)/np.linalg.norm(a)),
                 swot_reconstruction_relative=float(np.linalg.norm(b-v-f)/np.linalg.norm(b)),
                 sio_distinct_std_over_raw=float(e.std()/a.std()),swot_distinct_std_over_raw=float(f.std()/b.std()),
                 sio_common_energy_over_raw=float((u@u)/(a@a)),sio_distinct_energy_over_raw=float((e@e)/(a@a)),
                 sio_cross_energy_over_raw=float(2*(u@e)/(a@a)))
        for name,array in (('raw_sio',x1),('raw_swot',x2),('common',c1),('d1',d1),('d2',d2)):
            row[f'{name}_power_fraction_ge50km']=long_fraction(array,lat,lon)
        rows.append(row)
df=pd.DataFrame(rows)
output=ROOT/'analysis'/'cdpa_feature_audit'
output.mkdir(parents=True,exist_ok=True)
df.to_csv(output/'feature_diagnostics.csv',index=False,encoding='utf-8-sig')
print(df[['region','component','raw_corr','common_relative_difference','common_scale_swot_over_sio','common_scaled_residual','sio_distinct_std_over_raw','swot_distinct_std_over_raw']].to_string(index=False))
print('max reconstruction relative',df[['sio_reconstruction_relative','swot_reconstruction_relative']].max().max())
print('max standardized common difference',df['common_standardized_difference'].max())

print(df[['region','component','common_power_fraction_ge50km','d1_power_fraction_ge50km','d2_power_fraction_ge50km']].to_string(index=False))
