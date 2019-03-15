import pandas as pd
import numpy as np
import socket
import geoip2.database

reader = geoip2.database.Reader('GeoLite2-City.mmdb')

url_csv = pd.read_csv('top-10000.csv')

i = 0

urls = [line for line in url_csv['ip']]

ips = []
country_codes = []
latitudes = []
longitudes = []
i = 0
j = 0
for url in urls:
    try:
        location = reader.city(url).location
        latitudes.append(location.latitude)
        longitudes.append(location.longitude)
        ip = socket.gethostbyname(url)
        res = reader.country(ip)

        country_codes.append(res.country.iso_code)
        ips.append(ip)

    except:
        latitudes.append(np.nan)
        longitudes.append(np.nan)
        ips.append(np.nan)
        country_codes.append(np.nan)

    if not i % 100:
        print(j)
        i = 0

    i += 1
    j += 1

url_csv['ip'] = ips
url_csv['country'] = country_codes

url_csv['latitude'] = latitudes
url_csv['longitude'] = longitudes
url_csv.dropna(inplace=True)

url_csv.to_csv('top-10000.csv', index=False)
