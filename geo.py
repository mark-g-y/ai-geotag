import os

import googlemaps


gmaps = googlemaps.Client(key=os.getenv("GOOGLE_MAPS_API_KEY"))


def get_top_result(gmaps_reverse_geocode_results):
    location_type_priority = {
        'point_of_interest': 100000,
        'park': 100000,
        'airport': 100000,
        'natural_feature': 100000,
        'establishment': 90000,
        'premise': 90000,
        'subpremise': 90001,
        'street_address': 89999,
        'locality': 80000,
        # TODO: Add other location types.
        'default': 0
    }
    places_to_priority = []
    for result in gmaps_reverse_geocode_results:
        highest_priority_type = None
        for t in result['types']:
            if t in location_type_priority and (highest_priority_type is None or location_type_priority[highest_priority_type] < location_type_priority[t]):
                highest_priority_type = t
            elif t not in location_type_priority and highest_priority_type is None:
                highest_priority_type = 'default'
        places_to_priority.append(
            (result['place_id'], location_type_priority[highest_priority_type]))
    places_to_priority.sort(key=lambda x: x[1], reverse=True)
    print(places_to_priority)
    return places_to_priority[0][0]


def get_place_name(latitude, longitude):
    if latitude is None or longitude is None:
        return None

    reverse_geocode_results = gmaps.reverse_geocode((latitude, longitude))
    if not reverse_geocode_results:
        return None
    top_result_place_id = get_top_result(reverse_geocode_results)

    place_details = gmaps.place(top_result_place_id)

    return place_details['result']['name']


def get_lat_lng(place_name):
    geocode_results = gmaps.geocode(place_name)

    # Check if any results were returned
    if geocode_results:
        # The first result is usually the most relevant.
        # Extract latitude and longitude from the 'location' field of the geometry
        location = geocode_results[0]["geometry"]["location"]
        latitude = location["lat"]
        longitude = location["lng"]
        return (latitude, longitude)
    else:
        print(f"No results found for place name: '{place_name}'")
        return None
