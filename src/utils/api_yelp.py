import argparse
import json
import pprint
import requests
import sys
import urllib
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.parse import urlencode

API_KEY= 's-49w3tgPbYL9n7RpUPr211bpPJqRVYCDJmDhdHc3wTBaEnuvJlxJDC2C_m6-6DPG_0QR5CDsQ4em_6LPvss0QA60rckiO9wUf5JJ5R2fOAwys-5ApDKCgUjPeeSZXYx'

# API constants, you shouldn't have to change these.
API_HOST = 'https://api.yelp.com'
SEARCH_PATH = '/v3/businesses/search'
BUSINESS_PATH = '/v3/businesses/'  # Business ID will come after slash.

# Defaults for our simple example.
DEFAULT_TERM = 'dinner'
DEFAULT_LOCATION = 'San Francisco, CA'
SEARCH_LIMIT = 200

def request(host, path, api_key=API_KEY, url_params=None):
    """Given your API_KEY, send a GET request to the API.

    Args:
        host (str): The domain host of the API.
        path (str): The path of the API after the domain.
        API_KEY (str): Your API Key.
        url_params (dict): An optional set of query parameters in the request.

    Returns:
        dict: The JSON response from the request.

    Raises:
        HTTPError: An error occurs from the HTTP request.
    """
    url_params = url_params or {}
    url = '{0}{1}'.format(host, quote(path.encode('utf8')))
    headers = {
        'Authorization': 'Bearer %s' % api_key,
    }

    print(u'Querying {0} ...'.format(url))

    response = requests.request('GET', url, headers=headers, params=url_params)

    status_code = response.status_code
    print("Status Code:", status_code)

    if response.status_code != 200:
        print("Error occurred:", response.text)
        return None

    return response.json()


def search(term, location, api_key=API_KEY, limit=50, offset=0):
    """Query the Search API by a search term and location.

    Args:
        term (str): The search term passed to the API.
        location (str): The search location passed to the API.

    Returns:
        dict: The JSON response from the request.
    """

    url_params = {
        'term': term.replace(' ', '+'),
        'location': location.replace(' ', '+'),
        'limit': limit,
        'offset': offset
    }    

    return request(API_HOST, SEARCH_PATH, api_key, url_params=url_params)


def search_with_pagination(term, location, api_key=API_KEY, total=200):
    all_businesses = []
    for offset in range(0, total, 50):
        response = search(term, location, api_key, limit=min(50, total - offset), offset=offset)

        if not response or 'businesses' not in response:
            break  # Arrêter si la requête échoue ou si aucun autre résultat n'est disponible

        all_businesses.extend(response['businesses'])
        if len(response['businesses']) < 50:
            break  # Arrêter si moins de 50 résultats sont retournés dans la requête actuelle

    return all_businesses

def get_reviews(business_id, api_key=API_KEY):
    """Query the Business API for reviews of a specific business ID.

    Args:
        business_id (str): The ID of the business to query for reviews.
        api_key (str): Your API Key.

    Returns:
        dict: The JSON response containing reviews.
    """
    review_path = f'{BUSINESS_PATH}{business_id}/reviews'
    return request(API_HOST, review_path, api_key)


def get_review(business_id, api_key=API_KEY):
    """Query the Business API by a business ID.

    Args:
        business_id (str): The ID of the business to query.

    Returns:
        dict: The JSON response from the request.
    """
    review_path = f"{BUSINESS_PATH}{business_id}/reviews"
    return request(API_HOST, review_path, api_key)


def query_api(term, location):
    """Queries the API by the input values from the user.

    Args:
        term (str): The search term to query.
        location (str): The location of the business to query.
    """
    response = search(API_KEY, term, location)

    businesses = response.get('businesses')

    if not businesses:
        print(u'No businesses for {0} in {1} found.'.format(term, location))
        return

    business_id = businesses[0]['id']

    print(u'{0} businesses found, querying business info ' \
        'for the top result "{1}" ...'.format(
            len(businesses), business_id))
    response = get_business(API_KEY, business_id)

    print(u'Result for business "{0}" found:'.format(business_id))
    pprint.pprint(response, indent=2)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-q', '--term', dest='term', default=DEFAULT_TERM,
                        type=str, help='Search term (default: %(default)s)')
    parser.add_argument('-l', '--location', dest='location',
                        default=DEFAULT_LOCATION, type=str,
                        help='Search location (default: %(default)s)')

    input_values = parser.parse_args()

    try:
        query_api(input_values.term, input_values.location)
    except HTTPError as error:
        sys.exit(
            'Encountered HTTP error {0} on {1}:\n {2}\nAbort program.'.format(
                error.code,
                error.url,
                error.read(),
            )
        )

if __name__ == '__main__':
    main()