import pandas as pd
import numpy as np
import socket
import geoip2.database
import geoip2.errors
import json
import pprint as pp

reader = geoip2.database.Reader('GeoLite2-City.mmdb')

file = open('benign_ips_enriched.csv', 'w+')
file.close()

url_csv = pd.DataFrame()

with open('browsing_history.json') as file:
    data = json.load(file)

    urls_to_parse = []

    for line in data:
        try:
            urls_to_parse.append(line['url'])
        except:
            print('key "url" not defined in line {0}'.format(line))

    i = 0
    j = 0

    ips = []
    urls = []
    country_codes = []
    latitudes = []
    longitudes = []

    for url in urls_to_parse:

        try:
            ip = socket.gethostbyname(str(url).split('/')[2])
            country_code = reader.city(ip).country.iso_code
            latitude = reader.city(ip).location.latitude
            longitude = reader.city(ip).location.longitude

            ips.append(ip)
            country_codes.append(country_code)
            latitudes.append(latitude)
            longitudes.append(longitude)
            urls.append(url)
        except:
            urls.append(np.nan)
            latitudes.append(np.nan)
            longitudes.append(np.nan)
            ips.append(np.nan)
            country_codes.append(np.nan)

        if not i % 100:
            print(j)
            i = 0
        i += 1
        j += 1

    print("-" * 25)
    print(len(ips))
    print(len(country_codes))
    print(len(urls))
    print(len(latitudes))
    print(len(longitudes))

    url_csv['ip'] = ips
    url_csv['country'] = country_codes
    url_csv['url'] = urls
    url_csv['latitude'] = latitudes
    url_csv['longitude'] = longitudes

    url_csv.dropna(inplace=True)

    url_csv.to_csv('benign_ips_enriched.csv', index=False)
