import cdsapi

c = cdsapi.Client()

year_lst = [2020,2021,2022]

for year in year_lst:
    file_name = '45x45_'+ str(year) + '.nc'
    c.retrieve(
        'reanalysis-era5-single-levels',
        {
            'product_type': 'reanalysis',
            'format': 'netcdf',
            'variable': [
                '100m_u_component_of_wind', '10m_u_component_of_wind',
                '10m_v_component_of_wind',
            ],
            'year': [
                str(year),
            ],
        'month': [
                '01', '02', '03',
            '04', '05', '06',
            '07', '08', '09',
            '10',
            ],
        'day': [
            '01', '02', '03',
            '04', '05', '06',
            '07', '08', '09',
            '10',
        ],
            'time': [
                '00:00', '01:00', '02:00',
                '03:00', '04:00', '05:00',
                '06:00', '07:00', '08:00',
                '09:00', '10:00', '11:00',
                '12:00', '13:00', '14:00',
                '15:00', '16:00', '17:00',
                '18:00', '19:00', '20:00',
                '21:00', '22:00', '23:00',
            ],
            'area':[37+4.25, 125-4.25, 34.5-4.25, 127.5+4.25,]
            },
            './' + file_name)