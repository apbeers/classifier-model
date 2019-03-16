import pandas as pd
import numpy as np
import socket
import geoip2.database
import geoip2.errors
import json
import pprint as pp
import netaddr
reader = geoip2.database.Reader('GeoLite2-City.mmdb')


ip_csv = pd.read_csv('malicious_ips.csv', encoding="utf-8")

ips_to_parse = [line for line in ip_csv['ip']]

i = 0
j = 0

ips = []
country_codes = []
latitudes = []
longitudes = []

print(len(ips_to_parse))
for ip in ips_to_parse:

    try:
        ip = str(netaddr.IPAddress(ip, flags=netaddr.ZEROFILL).ipv4())
        country_code = reader.city(ip).country.iso_code
        latitude = reader.city(ip).location.latitude
        longitude = reader.city(ip).location.longitude

        latitudes.append(latitude)
        longitudes.append(longitude)
        ips.append(ip)
        country_codes.append(country_code)

    except:
        print('error')
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
print(len(latitudes))
print(len(longitudes))

ip_csv['ip'] = ips
ip_csv['country'] = country_codes
ip_csv['latitude'] = latitudes
ip_csv['longitude'] = longitudes


ip_csv.dropna(inplace=True)

ip_csv.to_csv('malicious_ips_locations.csv', index=False)
