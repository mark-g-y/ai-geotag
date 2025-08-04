from datetime import datetime

from PIL import Image, ExifTags


def coord_tuple_to_decimal(coord_tuple):
    return coord_tuple[0] + coord_tuple[1] / 60.0 + coord_tuple[2] / 3600.0


def coord_decimal_to_tuple(decimal_degrees):
    abs_decimal_degrees = abs(decimal_degrees)

    degrees = int(abs_decimal_degrees)
    minutes_float = (abs_decimal_degrees - degrees) * 60
    minutes = int(minutes_float)
    seconds_float = (minutes_float - minutes) * 60
    return (degrees, minutes, seconds_float)


def get_exif_data(file_path):
    img = Image.open(file_path)
    raw_exif_data = img.getexif()
    exif_data = {}
    exif_tags = {ExifTags.TAGS.get(
        tag, tag): value for tag, value in raw_exif_data.items()}
    for ifd_id in ExifTags.IFD:
        try:
            ifd = raw_exif_data.get_ifd(ifd_id)

            if ifd_id == ExifTags.IFD.GPSInfo:
                resolve = ExifTags.GPSTAGS
            else:
                resolve = ExifTags.TAGS

            exif_subcategory_data = {}
            for k, v in ifd.items():
                tag = resolve.get(k, k)
                exif_subcategory_data[tag] = v
            exif_data[ifd_id.name] = exif_subcategory_data
        except KeyError:
            pass
    return exif_data


def get_datetime(exif_data):
    if 'Exif' not in exif_data or 'DateTimeOriginal' not in exif_data['Exif']:
        return None

    date_time_original = exif_data['Exif']['DateTimeOriginal']
    offset_time_original = exif_data['Exif']['OffsetTimeOriginal']
    date_string = date_time_original + offset_time_original
    format_string = "%Y:%m:%d %H:%M:%S%z"
    dt_object = datetime.strptime(date_string, format_string)
    return dt_object


def get_lat_lng(exif_data):
    if 'GPSInfo' not in exif_data:
        return None, None

    gps_info = exif_data['GPSInfo']
    if 'GPSLatitude' not in gps_info or 'GPSLongitude' not in gps_info or 'GPSLatitudeRef' not in gps_info or 'GPSLongitudeRef' not in gps_info:
        return None, None

    latitude = (1 if gps_info['GPSLatitudeRef'] == 'N' else -1) * \
        coord_tuple_to_decimal(gps_info['GPSLatitude'])
    longitude = (1 if gps_info['GPSLongitudeRef'] == 'E' else -1) * \
        coord_tuple_to_decimal(gps_info['GPSLongitude'])

    return (latitude, longitude)


def lookup_exif_tag(tag_name):
    for tag, name in ExifTags.TAGS.items():
        if name == tag_name:
            return tag
    return None


def lookup_gps_exif_tag(tag_name):
    for tag, name in ExifTags.GPSTAGS.items():
        if name == tag_name:
            return tag
    return None


def set_coordinates_exif(filepath, lat, lng):
    lat_coord = coord_decimal_to_tuple(lat)
    lng_coord = coord_decimal_to_tuple(lng)
    print(lat_coord, lng_coord)
    image = Image.open(filepath)
    exif_dict = image.getexif() or {}
    if lookup_exif_tag('GPSInfo') not in exif_dict:
        exif_dict[lookup_exif_tag('GPSInfo')] = {}
    exif_dict[lookup_exif_tag('GPSInfo')][lookup_gps_exif_tag(
        'GPSLatitude')] = lat_coord
    exif_dict[lookup_exif_tag('GPSInfo')][lookup_gps_exif_tag(
        'GPSLatitudeRef')] = 'N' if lat >= 0 else 'S'
    exif_dict[lookup_exif_tag('GPSInfo')][lookup_gps_exif_tag(
        'GPSLongitude')] = lng_coord
    exif_dict[lookup_exif_tag('GPSInfo')][lookup_gps_exif_tag(
        'GPSLongitudeRef')] = 'E' if lng >= 0 else 'W'
    image.save(filepath, exif=exif_dict)
