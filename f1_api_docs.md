# F1 Forcaster API

# OpenF1 API (hogesnelheidstelemetrie & sessiedata)

(OpenF1-documentatie: Car data, Laps, Position, Intervals, Pit, Stints, Team radio, Weather, Race control, Sessions, Meetings, Location)

| Category | Feature(s) | API |
| --- | --- | --- |
| **Weather** | air_temperature, track_temperature, humidity, pressure, rainfall, wind_direction | OpenF1 |
| **Sessions** | session_key, session_name, date_start, date_end, circuit_short_name, session_type, year | OpenF1 |

## Weather

The weather over the track, updated every minute.

`curl "https://api.openf1.org/v1/weather?meeting_key=1208&wind_direction>=130&track_temperature>=52"`

> Output:
> 

`[
  {
    "air_temperature": 27.8,
    "date": "2023-05-07T18:42:25.233000+00:00",
    "humidity": 58,
    "meeting_key": 1208,
    "pressure": 1018.7,
    "rainfall": 0,
    "session_key": 9078,
    "track_temperature": 52.5,
    "wind_direction": 136,
    "wind_speed": 2.4
  }
]`

### HTTP Request

`GET https://api.openf1.org/v1/weather`

### Sample URL

[https://api.openf1.org/v1/weather?meeting_key=1208&wind_direction>=130&track_temperature>=52](https://api.openf1.org/v1/weather?meeting_key=1208&wind_direction%3E=130&track_temperature%3E=52)

### Attributes

| Name | Description |
| --- | --- |
| air_temperature | Air temperature (°C). |
| date | The UTC date and time, in ISO 8601 format. |
| humidity | Relative humidity (%). |
| meeting_key | The unique identifier for the meeting. Use `latest` to identify the latest or current meeting. |
| pressure | Air pressure (mbar). |
| rainfall | Whether there is rainfall. |
| session_key | The unique identifier for the session. Use `latest` to identify the latest or current session. |
| track_temperature | Track temperature (°C). |
| wind_direction | Wind direction (°), from 0° to 359°. |
| wind_speed | Wind speed (m/s). |

## Sessions

Provides information about sessions.
            A session refers to a distinct period of track activity 
during a Grand Prix or testing weekend (practice, qualifying, sprint, 
race, ...).

`curl "https://api.openf1.org/v1/sessions?country_name=Belgium&session_name=Sprint&year=2023"`

> Output:
> 

`[
  {
    "circuit_key": 7,
    "circuit_short_name": "Spa-Francorchamps",
    "country_code": "BEL",
    "country_key": 16,
    "country_name": "Belgium",
    "date_end": "2023-07-29T15:35:00+00:00",
    "date_start": "2023-07-29T15:05:00+00:00",
    "gmt_offset": "02:00:00",
    "location": "Spa-Francorchamps",
    "meeting_key": 1216,
    "session_key": 9140,
    "session_name": "Sprint",
    "session_type": "Race",
    "year": 2023
  }
]`

### HTTP Request

`GET https://api.openf1.org/v1/sessions`

### Sample URL

[https://api.openf1.org/v1/sessions?country_name=Belgium&session_name=Sprint&year=2023](https://api.openf1.org/v1/sessions?country_name=Belgium&session_name=Sprint&year=2023)

### Attributes

| Name | Description |
| --- | --- |
| circuit_key | The unique identifier for the circuit where the event takes place. |
| circuit_short_name | The short or common name of the circuit where the event takes place. |
| country_code | A code that uniquely identifies the country. |
| country_key | The unique identifier for the country where the event takes place. |
| country_name | The full name of the country where the event takes place. |
| date_end | The UTC ending date and time, in ISO 8601 format. |
| date_start | The UTC starting date and time, in ISO 8601 format. |
| gmt_offset | The difference in hours and minutes between local time at the location of the event and Greenwich Mean Time (GMT). |
| location | The city or geographical location where the event takes place. |
| meeting_key | The unique identifier for the meeting. Use `latest` to identify the latest or current meeting. |
| session_key | The unique identifier for the session. Use `latest` to identify the latest or current session. |
| session_name | The name of the session (`Practice 1`, `Qualifying`, `Race`, ...). |
| session_type | The type of the session (`Practice`, `Qualifying`, `Race`, ...). |
| year | The year the event takes place. |

# Jolpica F1 API (Ergast-compatibel historische data)

(Ergast-compatible endpoints: Seasons, Circuits, Races, Constructors, Drivers, Results, Sprint, Qualifying, DriverStandings, ConstructorStandings, Status)

| Category | Feature(s) | API |  |
| --- | --- | --- | --- |
| **Seasons** | season | Jolpica |  |
| **Circuits** | circuitId, circuitName, Location.lat, Location.long, locality, country | Jolpica |  |
| **Races** | raceName, date, time, round | Jolpica |  |
| **Constructors** | constructorId, name, nationality | Jolpica |  |
| **Drivers (stat.)** | driverId, givenName, familyName, dateOfBirth, nationality | Jolpica |  |
| **Results** | grid, position, laps, status, points, Time.millis/time, FastestLap.rank, FastestLap.lap, FastestLap.Time, FastestLap.AverageSpeed.speed | Jolpica |  |
| **Sprint results** | season, round, raceName, SprintResults.position, SprintResults.points | Jolpica |  |
| **Qualifying results** | Q1, Q2, Q3, position, Driver + Constructor identifiers | Jolpica |  |
| **Driver standings** | position, points, wins | Jolpica |  |
| **Constructor standings** | position, points, wins | Jolpica |  |
| **Status codes** | statusId, status description | Jolpica |  |

**Seasons**

Returns a list of seasons from earliest to latest.

**URL** : `/ergast/f1/seasons/`

[Available Query Parameters](https://github.com/jolpica/jolpica-f1/blob/main/docs/README.md#query-parameters)

---

**Route Parameters**

**Season**

Filters for a specified season. Year numbers are valid as is `current` to get the current season.

`/{season}/` -> ex: `/ergast/f1/2024/seasons/`

**Note**: To utilize the `season` parameter, it needs to be the first argument after `/ergast/f1/`.

---

**circuits**

Filters for only seasons featuring a specified circuit.

`/circuits/{circuitId}/` -> ex: `/ergast/f1/circuits/monza/seasons/`

---

**constructors**

Filters for only seasons featuring a specified constructor.

`/constructors/{constructorId}/` -> ex: `/ergast/f1/constructors/williams/seasons/`

---

**drivers**

Filters for only seasons featuring a specified driver.

`/drivers/{driverId}/` -> ex: `/ergast/f1/drivers/hamilton/seasons/`

---

**grid**

Filters for only seasons featuring a specified grid position.

`/grid/{gridPosition}/` -> ex: `/ergast/f1/grid/27/seasons/`

---

**status**

Filters for only seasons featuring a specified finishing status of a driver in at least one race that season.

`/status/{statusId}/` -> ex: `/ergast/f1/status/2/seasons/`

---

**Success Response**

**Code** : `200 OK`

**Response Fields** :

[Common Response Fields](https://github.com/jolpica/jolpica-f1/blob/main/docs/README.md#common-response-fields)

`MRData.SeasonTable` : The object containing the list of the all seasons returned.

`MRData.SeasonTable.Seasons` : The list of all seasons returned.

`MRData.SeasonTable.Seasons[i]` : A given season object.

---

**Season Object Fields:**

| Field | Always Included | Description | type |
| --- | --- | --- | --- |
| season | ✅ | Season year | String |
| url | ✅ | Wikipedia URL of the season | String |

---

**Examples:**

**Get list of seasons in F1 history**

`https://api.jolpi.ca/ergast/f1/seasons/`

`{
  "MRData": {
    "xmlns": "",
    "series": "f1",
    "url": "http://api.jolpi.ca/ergast/f1/seasons/",
    "limit": "30",
    "offset": "0",
    "total": "75",
    "SeasonTable": {
      "Seasons": [
        {
          "season": "1950",
          "url": "http://en.wikipedia.org/wiki/1950_Formula_One_season"
        },
        {
          "season": "1951",
          "url": "http://en.wikipedia.org/wiki/1951_Formula_One_season"
        },
        {
          "season": "1952",
          "url": "http://en.wikipedia.org/wiki/1952_Formula_One_season"
        },
        ...more
      ]
    }
  }
}`

**Circuits**

Returns a list of circuits in alphabetical order by `circuitId`

**URL** : `/ergast/f1/circuits/`

[Available Query Parameters](https://github.com/jolpica/jolpica-f1/blob/main/docs/README.md#query-parameters)

---

**Route Parameters**

**Season**

Filters only circuits which hosted a race in a given season. Year numbers are valid as is `current` to get the current season's list of circuits.

`/{season}/` -> ex: `/ergast/f1/2024/circuits/`

**Note**: To utilize the `season` parameter, it needs to be the first argument after `/ergast/f1/`.

---

**Round**

Filters only for the circuit that hosted the race in the specified round of the specific season. Round numbers 1 -> `n` races are valid as well as `last` and `next`.

`/{round}/` -> ex: `/ergast/f1/2024/1/circuits/`

**Note**: **Note**: To utilize the `round` parameter it needs to be used with the `season` filter and be the first argument after `/ergast/f1/{season}`.

---

**circuits**

Filters for only the circuit that matches the specified `circuitId`..

`/circuits/{circuitId}/` -> ex: `/ergast/f1/2024/circuits/albert_park/circuits/`

---

**constructors**

Filters for only circuits that the specified constructor has participated in a race at.

`/constructors/{constructorId}/` -> ex: `/ergast/f1/constructors/williams/circuits/`

---

**drivers**

Filters for only circuits that the specified driver has participated in a race at.

`/drivers/{driverId}/` -> ex: `/ergast/f1/drivers/hamilton/circuits/`

---

**fastest**

Filters
 for a list of circuits where a race finished with a driver completing a
 lap that was the ranked in the specified position.

`/fastest/{lapRank}/` -> ex: `/ergast/f1/fastest/24/circuits/`

---

**grid**

Filters for only circuits that have had a race with a specific grid position.

`/grid/{gridPosition}/` -> ex: `/ergast/f1/grid/29/circuits/`

---

**results**

Filters for only circuits that have had a race where a specific finishing position was valid.

`/results/{finishPosition}/` -> ex: `/ergast/f1/results/1/circuits/`

---

**status**

Filters for only circuits that have had a race where a driver finished with a specific `statusId`.

`/status/{statusId}/` -> ex: `/ergast/f1/status/2/drivers/`

---

**Success Response**

**Code** : `200 OK`

**Response Fields** :

[Common Response Fields](https://github.com/jolpica/jolpica-f1/blob/main/docs/README.md#common-response-fields)

`MRData.CircuitTable` : The object containing the list of the all drivers.

`MRData.CircuitTable.Circuits` : The list of all drivers returned.

`MRData.CircuitTable.Circuits[i]` : A given driver object.

---

**Circuits Object Fields:**

| Field | Always Included | Description | type |
| --- | --- | --- | --- |
| circuitId | ✅ | Unique ID of the circuit | String |
| url | ✅ | Wikipedia URL of circuit | String |
| circuitName | ✅ | Name of the Circuit | String |
| Location | ✅ | Location of circuit (lat, long, locality, country) | Object |

---

**Examples:**

**Get list of all circuits in F1 history**

`https://api.jolpi.ca/ergast/f1/circuits/`

```
{
  "MRData": {
    "xmlns": "",
    "series": "f1",
    "url": "http://api.jolpi.ca/ergast/f1/circuits/",
    "limit": "30",
    "offset": "0",
    "total": "77",
    "CircuitTable": {
      "Circuits": [
        {
          "circuitId": "adelaide",
          "url": "http://en.wikipedia.org/wiki/Adelaide_Street_Circuit",
          "circuitName": "Adelaide Street Circuit",
          "Location": {
            "lat": "-34.9272",
            "long": "138.617",
            "locality": "Adelaide",
            "country": "Australia"
          }
        },
        {
          "circuitId": "ain-diab",
          "url": "http://en.wikipedia.org/wiki/Ain-Diab_Circuit",
          "circuitName": "Ain Diab",
          "Location": {
            "lat": "33.5786",
            "long": "-7.6875",
            "locality": "Casablanca",
            "country": "Morocco"
          }
        },
        {
          "circuitId": "aintree",
          "url": "http://en.wikipedia.org/wiki/Aintree_Motor_Racing_Circuit",
          "circuitName": "Aintree",
          "Location": {
            "lat": "53.4769",
            "long": "-2.94056",
            "locality": "Liverpool",
            "country": "UK"
          }
        },
        ...more
      ]
    }
  }
}
```

**Get all circuits which have had a race with 29 driver results.**

`https://api.jolpi.ca/ergast/f1/results/29/circuits/`

```
{
  "MRData": {
    "xmlns": "",
    "series": "f1",
    "url": "http://api.jolpi.ca/ergast/f1/results/29/circuits/",
    "limit": "30",
    "offset": "0",
    "total": "1",
    "CircuitTable": {
      "position": "29",
      "Circuits": [
        {
          "circuitId": "indianapolis",
          "url": "http://en.wikipedia.org/wiki/Indianapolis_Motor_Speedway",
          "circuitName": "Indianapolis Motor Speedway",
          "Location": {
            "lat": "39.795",
            "long": "-86.2347",
            "locality": "Indianapolis",
            "country": "USA"
          }
        }
      ]
    }
  }
}
```

**Races**

Returns a list of races from earliest to latest.

**URL** : `/ergast/f1/races/`

[Available Query Parameters](https://github.com/jolpica/jolpica-f1/blob/main/docs/README.md#query-parameters)

---

**Route Parameters**

**Season**

Filters for races only from a specified season. Year numbers are valid as is `current` to get the current season.

`/{season}/` -> ex: `/ergast/f1/2024/races/`

**Note**: To utilize the `season` parameter, it needs to be the first argument after `/ergast/f1/`.

---

**Round**

Filters for the race for a specified round in a specific season. Round numbers 1 -> `n` races are valid as well as `last` and `next`.

`/{season}/{round}/` -> ex: `/ergast/f1/2024/5/races/`

**Note**: To utilize the `round` parameter it must be combined with a season filter and needs to be the first argument after `/ergast/f1/{season}/`.

---

**circuits**

Filters for only races featuring a specified circuit.

`/circuits/{circuitId}/` -> ex: `/ergast/f1/circuits/monza/races/`

---

**constructors**

Filters for only races featuring a specified constructor.

`/constructors/{constructorId}/` -> ex: `/ergast/f1/constructors/williams/races/`

---

**drivers**

Filters for only races featuring a specified driver.

`/drivers/{driverId}/` -> ex: `/ergast/f1/drivers/hamilton/races/`

---

**grid**

Filters for only races featuring a specified grid position.

`/grid/{gridPosition}/` -> ex: `/ergast/f1/grid/27/races/`

---

**status**

Filters for only races featuring a specified finishing status of a driver.

`/status/{statusId}/` -> ex: `/ergast/f1/status/2/races/`

---

**Success Response**

**Code** : `200 OK`

**Response Fields** :

[Common Response Fields](https://github.com/jolpica/jolpica-f1/blob/main/docs/README.md#common-response-fields)

`MRData.RaceTable` : The object containing the list of the all races.

`MRData.RaceTable.Races` : The list of all races returned.

`MRData.RaceTable.Races[i]` : A given race object.

---

**Race Object Fields:**

| Field | Always Included | Description | type |
| --- | --- | --- | --- |
| season | ✅ | Season year | String |
| round | ✅ | Round Number | String |
| url | 🟡 | Wikipedia URL of race | String |
| raceName | ✅ | Name of the race | String |
| Circuit | ✅ | Circuit information (circuitId, url, circuitName, Location) | Object |
| date | ✅ | Date of the race (YYYY-MM-DD) | String |
| time | 🟡 | UTC start time of the race | String |
| FirstPractice | 🟡 | First Practice (date, time) | Object |
| SecondPractice | 🟡 | Second Practice (date, time) | Object |
| ThirdPractice | 🟡 | Third Practice (date, time) | Object |
| Qualifying | 🟡 | Qualifying (date, time) | Object |
| Sprint | 🟡 | Sprint Race (date, time) | Object |
| SprintQualifying / SprintShootout | 🟡 | Shootouts took place in 2023, otherwise they are Qualifying (date, time) | Object |

---

**Examples:**

**Get list of races in F1 history**

`https://api.jolpi.ca/ergast/f1/races/`

```
{
  "MRData": {
    "xmlns": "",
    "series": "f1",
    "url": "http://api.jolpi.ca/ergast/f1/races/",
    "limit": "30",
    "offset": "0",
    "total": "1125",
    "RaceTable": {
      "Races": [
        {
          "season": "1950",
          "round": "1",
          "url": "http://en.wikipedia.org/wiki/1950_British_Grand_Prix",
          "raceName": "British Grand Prix",
          "Circuit": {
            "circuitId": "silverstone",
            "url": "http://en.wikipedia.org/wiki/Silverstone_Circuit",
            "circuitName": "Silverstone Circuit",
            "Location": {
              "lat": "52.0786",
              "long": "-1.01694",
              "locality": "Silverstone",
              "country": "UK"
            }
          },
          "date": "1950-05-13"
        },
        {
          "season": "1950",
          "round": "2",
          "url": "http://en.wikipedia.org/wiki/1950_Monaco_Grand_Prix",
          "raceName": "Monaco Grand Prix",
          "Circuit": {
            "circuitId": "monaco",
            "url": "http://en.wikipedia.org/wiki/Circuit_de_Monaco",
            "circuitName": "Circuit de Monaco",
            "Location": {
              "lat": "43.7347",
              "long": "7.42056",
              "locality": "Monte-Carlo",
              "country": "Monaco"
            }
          },
          "date": "1950-05-21"
        },
        ...more
      ]
    }
  }
}
```

**Get all races of a specific season (2024) and associated information**

`https://api.jolpi.ca/ergast/f1/2024/races/`

`{
  "MRData": {
    "xmlns": "",
    "series": "f1",
    "url": "http://api.jolpi.ca/ergast/f1/2024/races/",
    "limit": "30",
    "offset": "0",
    "total": "24",
    "RaceTable": {
      "season": "2024",
      "Races": [
        {
          "season": "2024",
          "round": "1",
          "url": "https://en.wikipedia.org/wiki/2024_Bahrain_Grand_Prix",
          "raceName": "Bahrain Grand Prix",
          "Circuit": {
            "circuitId": "bahrain",
            "url": "http://en.wikipedia.org/wiki/Bahrain_International_Circuit",
            "circuitName": "Bahrain International Circuit",
            "Location": {
              "lat": "26.0325",
              "long": "50.5106",
              "locality": "Sakhir",
              "country": "Bahrain"
            }
          },
          "date": "2024-03-02",
          "time": "15:00:00Z",
          "FirstPractice": {
            "date": "2024-02-29",
            "time": "11:30:00Z"
          },
          "SecondPractice": {
            "date": "2024-02-29",
            "time": "15:00:00Z"
          },
          "ThirdPractice": {
            "date": "2024-03-01",
            "time": "12:30:00Z"
          },
          "Qualifying": {
            "date": "2024-03-01",
            "time": "16:00:00Z"
          }
        },
        ...more
      ]
    }
  }
}`

**Constructors**

Returns a list of constructors alphabetically by `constructorId`

**URL** : `/ergast/f1/constructors/`

[Available Query Parameters](https://github.com/jolpica/jolpica-f1/blob/main/docs/README.md#query-parameters)

---

**Route Parameters**

**Season**

Filters only constructors that participated in a specified season. Year numbers are valid as is `current` to get the current season list of constructors.

`/{season}/` -> ex: `/ergast/f1/2024/constructors/`

**Note**: To utilize the `season` parameter, it needs to be the first argument after `/ergast/f1/`.

---

**Round**

Filters only constructors that participated in a specified round of a specific season. Round numbers 1 -> `n` races are valid as well as `last`.

`/{round}/` -> ex: `/ergast/f1/2024/1/constructors/`

**Note**: To utilize the `round` parameter it needs to be used with the `season` filter and be the first argument after `/ergast/f1/{season}/`

---

**circuits**

Filters for only constructors who have participated in a race at a given circuit.

`/circuits/{circuitId}/` -> ex: `/ergast/f1/circuits/bahrain/constructors/`

---

**constructors**

Filters for only a specified constructor.

`/constructors/{constructorId}/` -> ex: `/ergast/f1/constructors/williams/`

---

**drivers**

Filters for only constructors that had a driver race for them.

`/drivers/{driverId}/` -> ex: `/ergast/f1/drivers/hamilton/constructors/`

---

**fastest**

Filters for only constructors that finished a race with a lap that was the ranked in the specified position.

`/fastest/{lapRank}/` -> ex: `/ergast/f1/fastest/1/constructors/`

---

**grid**

Filters for only constructors which had a driver racing for them start a race in a specific grid position.

`/grid/{gridPosition}/` -> ex: `/ergast/f1/grid/1/constructors/`

---

**results**

Filters for only constructors which had a driver racing for them finish a race in a specific position.

`/results/{finishPosition}/` -> ex: `/ergast/f1/results/1/constructors/`

---

**status**

Filters for only constructors who had a driver finish a race with a specific `statusId`.

`/status/{statusId}/` -> ex: `/ergast/f1/status/2/constructors/`

---

**Success Response**

**Code** : `200 OK`

**Response Fields** :

[Common Response Fields](https://github.com/jolpica/jolpica-f1/blob/main/docs/README.md#common-response-fields)

`MRData.ConstructorTable` : The object containing the list of the all constructors.

`MRData.ConstructorTable.Constructors` : The list of all constructors returned.

`MRData.ConstructorTable.Constructors[i]` : A given constructor object.

---

**Constructor Object Fields:**

| Field | Always Included | Description | type |
| --- | --- | --- | --- |
| constructorId | 🟡 | Unique ID of the constructor | String |
| url | 🟡 | Wikipedia URL of the circuit | String |
| name | ✅ | Name of the Constructor | String |
| nationality | 🟡 | Nationality | String |

---

**Examples:**

**Get list of all constructors in F1 history**

`https://api.jolpi.ca/ergast/f1/constructors/`

```
{
  "MRData": {
    "xmlns": "",
    "series": "f1",
    "url": "http://api.jolpi.ca/ergast/f1/constructors/",
    "limit": "30",
    "offset": "0",
    "total": "212",
    "ConstructorTable": {
      "Constructors": [
        {
          "constructorId": "adams",
          "url": "http://en.wikipedia.org/wiki/Adams_(constructor)",
          "name": "Adams",
          "nationality": "American"
        },
        {
          "constructorId": "afm",
          "url": "http://en.wikipedia.org/wiki/Alex_von_Falkenhausen_Motorenbau",
          "name": "AFM",
          "nationality": "German"
        },
        ...more
      ]
    }
  }
}
```

**Get all constructors who had a driver who won a race**

`https://api.jolpi.ca/ergast/f1/results/1/constructors/`

`{
  "MRData": {
    "xmlns": "",
    "series": "f1",
    "url": "http://api.jolpi.ca/ergast/f1/results/1/constructors/",
    "limit": "30",
    "offset": "0",
    "total": "47",
    "ConstructorTable": {
      "position": "1",
      "Constructors": [
        {
          "constructorId": "alfa",
          "url": "http://en.wikipedia.org/wiki/Alfa_Romeo_in_Formula_One",
          "name": "Alfa Romeo",
          "nationality": "Swiss"
        },
        {
          "constructorId": "alphatauri",
          "url": "http://en.wikipedia.org/wiki/Scuderia_AlphaTauri",
          "name": "AlphaTauri",
          "nationality": "Italian"
        },
        ...more
      ]
    }
  }
}`

**Drivers**

Returns a list of drivers in alphabetical order by `driverId`

**URL** : `/ergast/f1/drivers/`

[Available Query Parameters](https://github.com/jolpica/jolpica-f1/blob/main/docs/README.md#query-parameters)

---

**Route Parameters**

**Season**

Filters only drivers that participated in a specified season. Year numbers are valid as is `current` to get the current season list of drivers.

`/{season}/` -> ex: `/ergast/f1/2024/drivers/`

**Note**: To utilize the `season` parameter, it needs to be the first argument after `/ergast/f1/`.

---

**Round**

Filters only drivers that participated in a specified round of a specific season. Round numbers 1 -> `n` races are valid as well as `last`.

`/{round}/` -> ex: `/ergast/f1/2024/1/drivers/`

**Note**: **Note**: To utilize the `round` parameter it needs to be used with the `season` filter and be the first argument after `/ergast/f1/{season}`.

---

**circuits**

Filters for only drivers who have participated in a race at a given circuit.

`/circuits/{circuitId}/` -> ex: `/ergast/f1/2024/circuits/albert_park/drivers/`

---

**constructors**

Filters for only drivers who have raced for a specified constructor.

`/constructors/{constructorId}/` -> ex: `/ergast/f1/constructors/williams/drivers/`

---

**drivers**

Filters for only drivers that match the specific `driverId`.

`/drivers/{driverId}/` -> ex: `/ergast/f1/drivers/hamilton/`

---

**fastest**

Filters for only drivers that finished a race with a lap that was the ranked in the specified position.

`/fastest/{lapRank}/` -> ex: `/ergast/f1/fastest/1/drivers/`

---

**grid**

Filters for only drivers who have started a race in a specific grid position.

`/grid/{gridPosition}/` -> ex: `/ergast/f1/grid/1/drivers/`

---

**results**

Filters for only drivers who have finished a race in a specific position.

`/results/{finishPosition}/` -> ex: `/ergast/f1/results/1/drivers/`

---

**status**

Filters for only drivers who have finished a race with a specific `statusId`.

`/status/{statusId}/` -> ex: `/ergast/f1/status/2/drivers/`

---

**Success Response**

**Code** : `200 OK`

**Response Fields** :

[Common Response Fields](https://github.com/jolpica/jolpica-f1/blob/main/docs/README.md#common-response-fields)

`MRData.DriverTable` : The object containing the list of the all drivers.

`MRData.DriverTable.Drivers` : The list of all drivers returned.

`MRData.DriverTable.Drivers[i]` : A given driver object.

---

**Driver Object Fields:**

| Field | Always Included | Description | type |
| --- | --- | --- | --- |
| driverId | ✅ | Unique ID of the Driver | String |
| permanentNumber | 🟡 | Permanent Number assigned to the driver | String |
| code | 🟡 | Driver Code, usually 3 characters | String |
| url | 🟡 | Wikipedia URL to the Drivers profile | String |
| givenName | ✅ | First name | String |
| familyName | ✅ | Last name | String |
| dateOfBirth | 🟡 | Date of Birth (YYYY-MM-DD format) | String |
| nationality | 🟡 | Nationality of Driver | String |

---

**Examples:**

**Get list of all drivers in F1 history**

`https://api.jolpi.ca/ergast/f1/drivers/`

```
{
  "MRData": {
    "xmlns": "",
    "series": "f1",
    "url": "http://api.jolpi.ca/ergast/f1/drivers/",
    "limit": "30",
    "offset": "0",
    "total": "860",
    "DriverTable": {
      "Drivers": [
        {
          "driverId": "abate",
          "url": "http://en.wikipedia.org/wiki/Carlo_Mario_Abate",
          "givenName": "Carlo",
          "familyName": "Abate",
          "dateOfBirth": "1932-07-10",
          "nationality": "Italian"
        },
        {
          "driverId": "abecassis",
          "url": "http://en.wikipedia.org/wiki/George_Abecassis",
          "givenName": "George",
          "familyName": "Abecassis",
          "dateOfBirth": "1913-03-21",
          "nationality": "British"
        },
        ...more
      ]
    }
  }
}
```

**Get all drivers who participated in the 2024 race at Albert Park**

- Note this is missing Logan Sargent as he did not start the race even though he participated in the weekend.

`https://api.jolpi.ca/ergast/f1/2024/circuits/albert_park/drivers/`

`{
  "MRData": {
    "xmlns": "",
    "series": "f1",
    "url": "http://api.jolpi.ca/ergast/f1/2024/circuits/albert_park/drivers/",
    "limit": "30",
    "offset": "0",
    "total": "19",
    "DriverTable": {
      "season": "2024",
      "circuitId": "albert_park",
      "Drivers": [
        {
          "driverId": "albon",
          "permanentNumber": "23",
          "code": "ALB",
          "url": "http://en.wikipedia.org/wiki/Alexander_Albon",
          "givenName": "Alexander",
          "familyName": "Albon",
          "dateOfBirth": "1996-03-23",
          "nationality": "Thai"
        },
        {
          "driverId": "alonso",
          "permanentNumber": "14",
          "code": "ALO",
          "url": "http://en.wikipedia.org/wiki/Fernando_Alonso",
          "givenName": "Fernando",
          "familyName": "Alonso",
          "dateOfBirth": "1981-07-29",
          "nationality": "Spanish"
        },
        ...more
      ]
    }
  }
}`

**Results**

Returns a list of race results.

**URL** : `/ergast/f1/results/`

[Available Query Parameters](https://github.com/jolpica/jolpica-f1/blob/main/docs/README.md#query-parameters)

---

**Route Parameters**

**Season**

Filters for race results only from a specified season. Year numbers are valid as is `current` to get the current season.

`/{season}/` -> ex: `/ergast/f1/2024/results/`

**Note**: To utilize the `season` parameter, it needs to be the first argument after `/ergast/f1/`.

---

**Round**

Filters for the race results for a specified round in a specific season. Round numbers 1 -> `n` races are valid as well as `last`.

`/{season}/{round}/` -> ex: `/ergast/f1/2024/last/results/`

**Note**: To utilize the `round` parameter it must be combined with a season filter and needs to be the first argument after `/ergast/f1/{season}/`.

---

**circuits**

Filters for only race results from races at a specified circuit.

`/circuits/{circuitId}/` -> ex: `/ergast/f1/circuits/monza/results/`

---

**constructors**

Filters for only race results for drivers racings for a specified constructor.

`/constructors/{constructorId}/` -> ex: `/ergast/f1/constructors/williams/results/`

---

**drivers**

Filters for only race results for a specified driver.

`/drivers/{driverId}/` -> ex: `/ergast/f1/drivers/hamilton/results/`

---

**fastest**

Filters for only race results of the driver who had the `n`th fastest lap of the race.

`/fastest/{lapRank}/` -> ex: `/ergast/f1/2024/fastest/1/results/`

---

**grid**

Filters for only race results for drivers starting in a specified grid position.

`/grid/{gridPosition}/` -> ex: `/ergast/f1/2024/grid/9/results/`

---

**status**

Filters for only race results of a driver who finished the race with a specific status.

`/status/{statusId}/` -> ex: `/ergast/f1/status/14/results/`

---

**Success Response**

**Code** : `200 OK`

**Response Fields** :

[Common Response Fields](https://github.com/jolpica/jolpica-f1/blob/main/docs/README.md#common-response-fields)

`MRData.RaceTable` : The object containing the list of the all races.

`MRData.RaceTable.Races` : The list of all races returned.

`MRData.RaceTable.Races[i]` : A given race object.

`MRData.RaceTable.Races[i].season` : The season the race is from.

`MRData.RaceTable.Races[i].round` : The round of season the race was.

`MRData.RaceTable.Races[i].url` : The wikipedia link of the race.

`MRData.RaceTable.Races[i].raceName` : The name of the race.

`MRData.RaceTable.Races[i].Circuit` : The circuit information.

`MRData.RaceTable.Races[i].date` : The date of the race.

`MRData.RaceTable.Races[i].time` : The time of the race.

`MRData.RaceTable.Races[i].Results` : The list of race results.

`MRData.RaceTable.Races[i].Results[j]` : The a given race result object.

---

**Race Result Object Fields:**

| Field | Always Included | Description | type |
| --- | --- | --- | --- |
| number | ✅ | The drivers number | String |
| position | ✅ | Finishing position of the driver | String |
| positionText | ✅ | Finishing position text representation | String |
| points | ✅ | Points the driver earned for this result | String |
| Driver | ✅ | Driver information (driverId, permanentNumber, code, url, givenName, familyName, dateOfBirth, nationality) | Object |
| Constructor | 🟡 | Driver information (constructorId, url, name, nationality) | Object |
| grid | 🟡 | The driver's grid position | String |
| laps | 🟡 | The laps this driver completed | String |
| status | 🟡 | The drivers finishing status in long form | String |
| FastestLap | 🟡 | The fastest lap information for this driver (rank, lap, Time, AverageSpeed) | Object |

---

**Examples:**

**Get the race results from the 9th round of the 2021 season.**

`http://api.jolpi.ca/ergast/f1/2021/9/results/`

```
{
  "MRData": {
    "xmlns": "",
    "series": "f1",
    "url": "http://api.jolpi.ca/ergast/f1/2021/9/results/",
    "limit": "30",
    "offset": "0",
    "total": "20",
    "RaceTable": {
      "season": "2021",
      "round": "9",
      "Races": [
        {
          "season": "2021",
          "round": "9",
          "url": "http://en.wikipedia.org/wiki/2021_Austrian_Grand_Prix",
          "raceName": "Austrian Grand Prix",
          "Circuit": {
            "circuitId": "red_bull_ring",
            "url": "http://en.wikipedia.org/wiki/Red_Bull_Ring",
            "circuitName": "Red Bull Ring",
            "Location": {
              "lat": "47.2197",
              "long": "14.7647",
              "locality": "Spielberg",
              "country": "Austria"
            }
          },
          "date": "2021-07-04",
          "time": "13:00:00Z",
          "Results": [
            {
              "number": "33",
              "position": "1",
              "positionText": "1",
              "points": "26",
              "Driver": {
                "driverId": "max_verstappen",
                "permanentNumber": "33",
                "code": "VER",
                "url": "http://en.wikipedia.org/wiki/Max_Verstappen",
                "givenName": "Max",
                "familyName": "Verstappen",
                "dateOfBirth": "1997-09-30",
                "nationality": "Dutch"
              },
              "Constructor": {
                "constructorId": "red_bull",
                "url": "http://en.wikipedia.org/wiki/Red_Bull_Racing",
                "name": "Red Bull",
                "nationality": "Austrian"
              },
              "grid": "1",
              "laps": "71",
              "status": "Finished",
              "Time": {
                "millis": "5034543",
                "time": "1:23:54.543"
              },
              "FastestLap": {
                "rank": "1",
                "lap": "62",
                "Time": {
                  "time": "1:06.200"
                },
                "AverageSpeed": {
                  "units": "kph",
                  "speed": "234.815"
                }
              }
            },
            {
              "number": "77",
              "position": "2",
              "positionText": "2",
              "points": "18",
              "Driver": {
                "driverId": "bottas",
                "permanentNumber": "77",
                "code": "BOT",
                "url": "http://en.wikipedia.org/wiki/Valtteri_Bottas",
                "givenName": "Valtteri",
                "familyName": "Bottas",
                "dateOfBirth": "1989-08-28",
                "nationality": "Finnish"
              },
              "Constructor": {
                "constructorId": "mercedes",
                "url": "http://en.wikipedia.org/wiki/Mercedes-Benz_in_Formula_One",
                "name": "Mercedes",
                "nationality": "German"
              },
              "grid": "5",
              "laps": "71",
              "status": "Finished",
              "Time": {
                "millis": "5052516",
                "time": "+17.973"
              },
              "FastestLap": {
                "rank": "6",
                "lap": "52",
                "Time": {
                  "time": "1:08.374"
                },
                "AverageSpeed": {
                  "units": "kph",
                  "speed": "227.349"
                }
              }
            },
            ...more results from the race
          ]
        },
        ...more races
      ]
    }
  }
}
```

**Get all 2023 race results of all races where a driver finished 2 laps down**

`https://api.jolpi.ca/ergast/f1/2023/status/12/results/`

`{
  "MRData": {
    "xmlns": "",
    "series": "f1",
    "url": "http://api.jolpi.ca/ergast/f1/2023/status/12/results/",
    "limit": "30",
    "offset": "0",
    "total": "6",
    "RaceTable": {
      "season": "2023",
      "status": "12",
      "Races": [
        {
          "season": "2023",
          "round": "1",
          "url": "https://en.wikipedia.org/wiki/2023_Bahrain_Grand_Prix",
          "raceName": "Bahrain Grand Prix",
          "Circuit": {
            "circuitId": "bahrain",
            "url": "http://en.wikipedia.org/wiki/Bahrain_International_Circuit",
            "circuitName": "Bahrain International Circuit",
            "Location": {
              "lat": "26.0325",
              "long": "50.5106",
              "locality": "Sakhir",
              "country": "Bahrain"
            }
          },
          "date": "2023-03-05",
          "time": "15:00:00Z",
          "Results": [
            {
              "number": "4",
              "position": "17",
              "positionText": "17",
              "points": "0",
              "Driver": {
                "driverId": "norris",
                "permanentNumber": "4",
                "code": "NOR",
                "url": "http://en.wikipedia.org/wiki/Lando_Norris",
                "givenName": "Lando",
                "familyName": "Norris",
                "dateOfBirth": "1999-11-13",
                "nationality": "British"
              },
              "Constructor": {
                "constructorId": "mclaren",
                "url": "http://en.wikipedia.org/wiki/McLaren",
                "name": "McLaren",
                "nationality": "British"
              },
              "grid": "11",
              "laps": "55",
              "status": "+2 Laps",
              "FastestLap": {
                "rank": "3",
                "lap": "51",
                "Time": {
                  "time": "1:35.822"
                },
                "AverageSpeed": {
                  "units": "kph",
                  "speed": "203.327"
                }
              }
            }
          ]
        },
        {
          "season": "2023",
          "round": "6",
          "url": "https://en.wikipedia.org/wiki/2023_Monaco_Grand_Prix",
          "raceName": "Monaco Grand Prix",
          "Circuit": {
            "circuitId": "monaco",
            "url": "http://en.wikipedia.org/wiki/Circuit_de_Monaco",
            "circuitName": "Circuit de Monaco",
            "Location": {
              "lat": "43.7347",
              "long": "7.42056",
              "locality": "Monte-Carlo",
              "country": "Monaco"
            }
          },
          "date": "2023-05-28",
          "time": "13:00:00Z",
          "Results": [
            {
              "number": "22",
              "position": "15",
              "positionText": "15",
              "points": "0",
              "Driver": {
                "driverId": "tsunoda",
                "permanentNumber": "22",
                "code": "TSU",
                "url": "http://en.wikipedia.org/wiki/Yuki_Tsunoda",
                "givenName": "Yuki",
                "familyName": "Tsunoda",
                "dateOfBirth": "2000-05-11",
                "nationality": "Japanese"
              },
              "Constructor": {
                "constructorId": "alphatauri",
                "url": "http://en.wikipedia.org/wiki/Scuderia_AlphaTauri",
                "name": "AlphaTauri",
                "nationality": "Italian"
              },
              "grid": "9",
              "laps": "76",
              "status": "+2 Laps",
              "FastestLap": {
                "rank": "16",
                "lap": "36",
                "Time": {
                  "time": "1:17.680"
                },
                "AverageSpeed": {
                  "units": "kph",
                  "speed": "154.649"
                }
              }
            },
            ...more drivers 2 laps down in Monaco 
          ]
        },
        ...more races in 2023 where a driver finished 2 laps down
      ]
    }
  }
}`

**Sprint**

Returns a list of sprint race results.

**URL** : `/ergast/f1/sprint/`

[Available Query Parameters](https://github.com/jolpica/jolpica-f1/blob/main/docs/README.md#query-parameters)

---

**Route Parameters**

**Season**

Filters for sprint results only from a specified season. Year numbers are valid, as is `current`.

`/{season}/` -> ex: `/ergast/f1/2023/sprint/`

**Note**: To utilize the `season` parameter, it needs to be the first argument after `/ergast/f1/`.

---

**Round**

Filters for the sprint results for a specified round in a specific season. Round numbers 1 -> `n` races are valid as well as `last`.

`/{season}/{round}/` -> ex: `/ergast/f1/2024/5/sprint/`

**Note**: To utilize the `round` parameter it must be combined with a season filter and needs to be the first argument after `/ergast/f1/{season}/`.

---

**circuits**

Filters for only sprint results from races at a specified circuit.

`/circuits/{circuitId}/` -> ex: `/ergast/f1/circuits/red_bull_ring/sprint/`

---

**constructors**

Filters for only sprint results for drivers racing for a specified constructor.

`/constructors/{constructorId}/` -> ex: `/ergast/f1/constructors/mclaren/sprint/`

---

**drivers**

Filters for only sprint results for a specified driver.

`/drivers/{driverId}/` -> ex: `/ergast/f1/drivers/norris/sprint/`

---

**grid**

Filters for only sprint results for drivers starting the sprint in a specified grid position.

`/grid/{gridPosition}/` -> ex: `/ergast/f1/2024/grid/1/sprint/`

---

**status**

Filters for only sprint results of a driver who finished the sprint with a specific status.

`/status/{statusId}/` -> ex: `/ergast/f1/status/1/sprint/`

---

**Success Response**

**Code** : `200 OK`

**Response Fields** :

[Common Response Fields](https://github.com/jolpica/jolpica-f1/blob/main/docs/README.md#common-response-fields)

`MRData.RaceTable` : The object containing the list of the all races with sprints matching filters.

`MRData.RaceTable.Races` : The list of all races returned.

`MRData.RaceTable.Races[i]` : A given race object.

`MRData.RaceTable.Races[i].season` : The season the race is from.

`MRData.RaceTable.Races[i].round` : The round of season the race was.

`MRData.RaceTable.Races[i].url` : The wikipedia link of the race.

`MRData.RaceTable.Races[i].raceName` : The name of the race.

`MRData.RaceTable.Races[i].Circuit` : The circuit information.

`MRData.RaceTable.Races[i].date` : The date of the main race.

`MRData.RaceTable.Races[i].time` : The time of the main race.

`MRData.RaceTable.Races[i].SprintResults` : The list of sprint race results.

`MRData.RaceTable.Races[i].SprintResults[j]` : The a given sprint result object.

---

**Sprint Result Object Fields:**

| Field | Always Included | Description | type |
| --- | --- | --- | --- |
| number | ✅ | The drivers number | String |
| position | ✅ | Finishing position of the driver | String |
| positionText | ✅ | Finishing position text representation | String |
| points | ✅ | Points the driver earned for this sprint result | String |
| Driver | ✅ | Driver information (driverId, permanentNumber, code, url, givenName, familyName, dateOfBirth, nationality) | Object |
| Constructor | 🟡 | Constructor information (constructorId, url, name, nationality) | Object |
| grid | 🟡 | The driver's grid position for the sprint | String |
| laps | 🟡 | The laps this driver completed in the sprint | String |
| status | 🟡 | The drivers finishing status in the sprint | String |
| Time | 🟡 | Finishing time information (millis, time) | Object |
| FastestLap | 🟡 | The fastest lap information for this driver (rank, lap, Time, AverageSpeed) | Object |

---

**Examples:**

**Get the sprint results from the 5th round (China) of the 2024 season.**

`http://api.jolpi.ca/ergast/f1/2024/5/sprint/`

```
{
    "MRData": {
        "xmlns": "",
        "series": "f1",
        "url": "http://api.jolpi.ca/ergast/f1/2024/5/sprint/",
        "limit": "30",
        "offset": "0",
        "total": "20",
        "RaceTable": {
            "season": "2024",
            "round": "5",
            "Races": [
                {
                    "season": "2024",
                    "round": "5",
                    "url": "https://en.wikipedia.org/wiki/2024_Chinese_Grand_Prix",
                    "raceName": "Chinese Grand Prix",
                    "Circuit": {
                        "circuitId": "shanghai",
                        "url": "https://en.wikipedia.org/wiki/Shanghai_International_Circuit",
                        "circuitName": "Shanghai International Circuit",
                        "Location": {
                            "lat": "31.3389",
                            "long": "121.22",
                            "locality": "Shanghai",
                            "country": "China"
                        }
                    },
                    "date": "2024-04-21",
                    "time": "07:00:00Z",
                    "SprintResults": [
                        {
                            "number": "1",
                            "position": "1",
                            "positionText": "1",
                            "points": "8",
                            "Driver": {
                                "driverId": "max_verstappen",
                                "permanentNumber": "33",
                                "code": "VER",
                                "url": "http://en.wikipedia.org/wiki/Max_Verstappen",
                                "givenName": "Max",
                                "familyName": "Verstappen",
                                "dateOfBirth": "1997-09-30",
                                "nationality": "Dutch"
                            },
                            "Constructor": {
                                "constructorId": "red_bull",
                                "url": "http://en.wikipedia.org/wiki/Red_Bull_Racing",
                                "name": "Red Bull",
                                "nationality": "Austrian"
                            },
                            "grid": "4",
                            "laps": "19",
                            "status": "Finished",
                            "Time": {
                                "millis": "1924660",
                                "time": "32:04.660"
                            },
                            "FastestLap": {
                                "rank": "1",
                                "lap": "3",
                                "Time": {
                                    "time": "1:40.331"
                                }
                            }
                        },
                        ...more results from the sprint
                    ]
                }
            ]
        }
    }
}
```

**Get all sprint results for Lando Norris in the 2023 season**

`https://api.jolpi.ca/ergast/f1/2023/drivers/norris/sprint/`

`{
    "MRData": {
        "xmlns": "",
        "series": "f1",
        "url": "http://api.jolpi.ca/ergast/f1/2023/drivers/norris/sprint/",
        "limit": "30",
        "offset": "0",
        "total": "6",
        "RaceTable": {
            "season": "2023",
            "driverId": "norris",
            "Races": [
                {
                    "season": "2023",
                    "round": "4",
                    "url": "https://en.wikipedia.org/wiki/2023_Azerbaijan_Grand_Prix",
                    "raceName": "Azerbaijan Grand Prix",
                    // ... Circuit info ...
                    "date": "2023-04-30",
                    "time": "11:00:00Z",
                    "SprintResults": [
                        {
                            "number": "4",
                            "position": "17",
                            "positionText": "17",
                            "points": "0",
                            "Driver": {
                                "driverId": "norris",
                                "permanentNumber": "4",
                                "code": "NOR",
                                "url": "http://en.wikipedia.org/wiki/Lando_Norris",
                                "givenName": "Lando",
                                "familyName": "Norris",
                                "dateOfBirth": "1999-11-13",
                                "nationality": "British"
                            },
                            "Constructor": {
                                "constructorId": "mclaren",
                                "url": "http://en.wikipedia.org/wiki/McLaren",
                                "name": "McLaren",
                                "nationality": "British"
                            },
                            "grid": "10",
                            "laps": "17",
                            "status": "Finished",
                            "Time": {
                                "millis": "2048771",
                                "time": "+51.104"
                            },
                            "FastestLap": {
                                "lap": "15",
                                "Time": {
                                    "time": "1:44.484"
                                }
                            }
                        }
                    ]
                },
                {
                    "season": "2023",
                    "round": "10",
                    "url": "https://en.wikipedia.org/wiki/2023_Austrian_Grand_Prix",
                    "raceName": "Austrian Grand Prix",
                    // ... Circuit info ...
                    "date": "2023-07-02",
                    "time": "13:00:00Z",
                    "SprintResults": [
                        // Sprint results
                    ]
                },
                 // ... Results from Belgium, Qatar, USA, Brazil sprints ...
            ]
        }
    }
}`

**Qualifying**

Returns a list of qualification results from each race.

**URL** : `/ergast/f1/qualifying/`

[Available Query Parameters](https://github.com/jolpica/jolpica-f1/blob/main/docs/README.md#query-parameters)

---

**Route Parameters**

**Season**

Filters for qualifying results only from a specified season. Year numbers are valid as is `current` to get the current season.

`/{season}/` -> ex: `/ergast/f1/2024/qualifying/`

**Note**: To utilize the `season` parameter, it needs to be the first argument after `/ergast/f1/`.

---

**Round**

Filters for the qualifying results for a specified round in a specific season. Round numbers 1 -> `n` races are valid as well as `last` and `next`.

`/{season}/{round}/` -> ex: `/ergast/f1/2024/5/qualifying/`

**Note**: To utilize the `round` parameter it must be combined with a season filter and needs to be the first argument after `/ergast/f1/{season}/`.

---

**circuits**

Filters for the qualifying results at a specified circuit.

`/circuits/{circuitId}/` -> ex: `/ergast/f1/circuits/monza/qualifying/`

---

**constructors**

Filters for the qualifying results of drivers driving for a specified constructor.

`/constructors/{constructorId}/` -> ex: `/ergast/f1/constructors/williams/qualifying/`

---

**drivers**

Filters for the qualifying results of a specified driver.

`/drivers/{driverId}/` -> ex: `/ergast/f1/drivers/hamilton/qualifying/`

---

**grid**

Filters for the qualifying results of a driver who started the associated race in a specified grid position.

`/grid/{gridPosition}/` -> ex: `/ergast/f1/grid/18/qualifying/`

**Example**: Pierre Gasly finished Qualifying
 in 13th for the 2024 Azerbaijan Grand Prix but was disqualified due to 
instantaneous fuel mass flow limit and started P18 after Ocon and 
Hamilton. So `/ergast/f1/2024/17/grid/18/qualifying/` would return Gasly's results instead of Zhou who finished Qualifying in P18.

---

**fastest**

Filters for the qualifying results a driver with the fastest lap rank at a given Grand Prix.

`/fastest/{lapRank}/` -> ex: `/ergast/f1/fastest/2/qualifying/`

**Example**: If you do `/ergast/f1/2024/17/fastest/1/qualifying/`
 it will return Lando Norris' qualifying result for the 2024 Azerbaijan 
Grand Prix as he had the fastest lap in the race, even though he 
qualified 16th.

---

**status**

Filters for the qualifying results of any drivers with the finishing statusId at a given Grand Prix.

`/status/{statusId}/` -> ex: `/ergast/f1/status/11/qualifying/`

---

**Success Response**

**Code** : `200 OK`

**Response Fields** :

[Common Response Fields](https://github.com/jolpica/jolpica-f1/blob/main/docs/README.md#common-response-fields)

`MRData.RaceTable` : The object containing the list of the all races and associated filters.

`MRData.RaceTable.Races` : The list of all races returned.

`MRData.RaceTable.Races[i]` : A given race object.

`MRData.RaceTable.Races[i].QualifyingResults` : The list of qualifying results for the given race.

`MRData.RaceTable.Races[i].QualifyingResults[i]` : A given qualifying result object.

---

**Qualifying Result Object Fields:**

| Field | Always Included | Description | type |
| --- | --- | --- | --- |
| number | ✅ | Driver's car number | String |
| position | 🟡 | Qualifying Result Position | String |
| Driver | ✅ | Driver information (driverId, permanentNumber, code, url, givenName, familyName, dateOfBirth, nationality) | Object |
| Constructor | ✅ | Constructor information (constructorId, url, name, nationality) | Object |
| Q1 | 🟡 | Qualifying 1 Result (mm:ss.sss) | String |
| Q2 | 🟡 | Qualifying 2 Result (mm:ss.sss) | String |
| Q3 | 🟡 | Qualifying 3 Result (mm:ss.sss) | String |

---

**Examples:**

**Get list of all qualifying results in 2024**

`http://api.jolpi.ca/ergast/f1/2024/qualifying/`

```
{
  "MRData": {
    "xmlns": "",
    "series": "f1",
    "url": "http://api.jolpi.ca/ergast/f1/2024/qualifying/",
    "limit": "30",
    "offset": "0",
    "total": "359",
    "RaceTable": {
      "season": "2024",
      "Races": [
        {
          "season": "2024",
          "round": "1",
          "url": "https://en.wikipedia.org/wiki/2024_Bahrain_Grand_Prix",
          "raceName": "Bahrain Grand Prix",
          "Circuit": {
            "circuitId": "bahrain",
            "url": "http://en.wikipedia.org/wiki/Bahrain_International_Circuit",
            "circuitName": "Bahrain International Circuit",
            "Location": {
              "lat": "26.0325",
              "long": "50.5106",
              "locality": "Sakhir",
              "country": "Bahrain"
            }
          },
          "date": "2024-03-02",
          "time": "15:00:00Z",
          "QualifyingResults": [
            {
              "number": "1",
              "position": "1",
              "Driver": {
                "driverId": "max_verstappen",
                "permanentNumber": "33",
                "code": "VER",
                "url": "http://en.wikipedia.org/wiki/Max_Verstappen",
                "givenName": "Max",
                "familyName": "Verstappen",
                "dateOfBirth": "1997-09-30",
                "nationality": "Dutch"
              },
              "Constructor": {
                "constructorId": "red_bull",
                "url": "http://en.wikipedia.org/wiki/Red_Bull_Racing",
                "name": "Red Bull",
                "nationality": "Austrian"
              },
              "Q1": "1:30.031",
              "Q2": "1:29.374",
              "Q3": "1:29.179"
            },
            {
              "number": "16",
              "position": "2",
              "Driver": {
                "driverId": "leclerc",
                "permanentNumber": "16",
                "code": "LEC",
                "url": "http://en.wikipedia.org/wiki/Charles_Leclerc",
                "givenName": "Charles",
                "familyName": "Leclerc",
                "dateOfBirth": "1997-10-16",
                "nationality": "Monegasque"
              },
              "Constructor": {
                "constructorId": "ferrari",
                "url": "http://en.wikipedia.org/wiki/Scuderia_Ferrari",
                "name": "Ferrari",
                "nationality": "Italian"
              },
              "Q1": "1:30.243",
              "Q2": "1:29.165",
              "Q3": "1:29.407"
            },
            ...more
          ]
        }
      ]
    }
  }
}
```

**Get the qualifying results from the 17th Round of the 2024 season.**

`http://api.jolpi.ca/ergast/f1/2024/17/qualifying/`

`{
  "MRData": {
    "xmlns": "",
    "series": "f1",
    "url": "http://api.jolpi.ca/ergast/f1/2024/17/qualifying/",
    "limit": "30",
    "offset": "0",
    "total": "20",
    "RaceTable": {
      "season": "2024",
      "round": "17",
      "Races": [
        {
          "season": "2024",
          "round": "17",
          "url": "https://en.wikipedia.org/wiki/2024_Azerbaijan_Grand_Prix",
          "raceName": "Azerbaijan Grand Prix",
          "Circuit": {
            "circuitId": "baku",
            "url": "http://en.wikipedia.org/wiki/Baku_City_Circuit",
            "circuitName": "Baku City Circuit",
            "Location": {
              "lat": "40.3725",
              "long": "49.8533",
              "locality": "Baku",
              "country": "Azerbaijan"
            }
          },
          "date": "2024-09-15",
          "time": "11:00:00Z",
          "QualifyingResults": [
            {
              "number": "16",
              "position": "1",
              "Driver": {
                "driverId": "leclerc",
                "permanentNumber": "16",
                "code": "LEC",
                "url": "http://en.wikipedia.org/wiki/Charles_Leclerc",
                "givenName": "Charles",
                "familyName": "Leclerc",
                "dateOfBirth": "1997-10-16",
                "nationality": "Monegasque"
              },
              "Constructor": {
                "constructorId": "ferrari",
                "url": "http://en.wikipedia.org/wiki/Scuderia_Ferrari",
                "name": "Ferrari",
                "nationality": "Italian"
              },
              "Q1": "1:42.775",
              "Q2": "1:42.056",
              "Q3": "1:41.365"
            },
            {
              "number": "81",
              "position": "2",
              "Driver": {
                "driverId": "piastri",
                "permanentNumber": "81",
                "code": "PIA",
                "url": "http://en.wikipedia.org/wiki/Oscar_Piastri",
                "givenName": "Oscar",
                "familyName": "Piastri",
                "dateOfBirth": "2001-04-06",
                "nationality": "Australian"
              },
              "Constructor": {
                "constructorId": "mclaren",
                "url": "http://en.wikipedia.org/wiki/McLaren",
                "name": "McLaren",
                "nationality": "British"
              },
              "Q1": "1:43.033",
              "Q2": "1:42.598",
              "Q3": "1:41.686"
            },
            ...more
          ]
        }
      ]
    }
  }
}`

**Driver Standings**

Returns a season's drivers standings from first to last place.

**URL** : `/ergast/f1/{season}/driverstandings/`

[Available Query Parameters](https://github.com/jolpica/jolpica-f1/blob/main/docs/README.md#query-parameters)

---

**Route Parameters**

**Season (required)**

Filters for the drivers standing of a specified season. Year numbers are valid as is `current` to get the current seasons drivers standings.

`/{season}/` -> ex: `/ergast/f1/2024/driverstandings/`

**Note**: To utilize the `season` parameter, it needs to be the first argument after `/ergast/f1/`.

---

**Round**

Filters for the drivers standings after a specified round in a specific season. Round numbers 1 -> `n` races are valid as well as `last`.

`/{season}/{round}/` -> ex: `/ergast/f1/2024/5/driverstandings/`

**Note**: To utilize the `round` parameter it must be combined with a season filter and needs to be the first argument after `/ergast/f1/{season}/`.

---

**drivers**

Filters for only a specific driver's drivers standing information for a given year.

`/drivers/{driverId}/` -> ex: `/ergast/f1/2024/drivers/hamilton/driverstandings/`

---

**position**

Filters for only the driver in a given position for a given year.

`/{finishPosition}` -> ex: `/ergast/f1/2024/driverstandings/1/`

**Note**: The position must be at the end after any filters and after `/driverstandings/`

---

**Success Response**

**Code** : `200 OK`

**Response Fields** :

[Common Response Fields](https://github.com/jolpica/jolpica-f1/blob/main/docs/README.md#common-response-fields)

`MRData.StandingsTable` : The object containing the season's drivers standing information.

`MRData.StandingsTable.season` : The filtered season.

`MRData.StandingsTable.round` : The round that the season the standings represent.

`MRData.StandingsTable.StandingsLists` : The list of drivers standings.

`MRData.StandingsTable.StandingsLists[i]` : A given drivers standings list object.

`MRData.StandingsTable.StandingsLists[i].DriverStandings` : The list of drivers standings objects.

---

**Drivers Standing Object Fields:**

| Field | Always Included | Description | type |
| --- | --- | --- | --- |
| position | 🟡 | Position in the Championship | String |
| positonText | ✅ | Description of position `*` | String |
| points | ✅ | Total points in the Championship | String |
| wins | ✅ | Count of race wins | String |
| Driver | ✅ | Driver information (driverId, url, givenName, familyName, dateOfBirth, nationality) | Object |
| Constructors | ✅ | List of all constructors the driver drove for in the given season | Array |
- - Possible values for positionText include: `E` Excluded, `D` Disqualified (1997 Schumacher),  for ineligible or the position as a string otherwise.

---

**Examples:**

**Get the 1972 season's drivers standing information**

`https://api.jolpi.ca/ergast/f1/1972/driverstandings/`

```
{
  "MRData": {
    "xmlns": "",
    "series": "f1",
    "url": "http://api.jolpi.ca/ergast/f1/1972/driverstandings/",
    "limit": "30",
    "offset": "0",
    "total": "42",
    "StandingsTable": {
      "season": "1972",
      "round": "12",
      "StandingsLists": [
        {
          "season": "1972",
          "round": "12",
          "DriverStandings": [
            {
              "position": "1",
              "positionText": "1",
              "points": "61",
              "wins": "5",
              "Driver": {
                "driverId": "emerson_fittipaldi",
                "url": "http://en.wikipedia.org/wiki/Emerson_Fittipaldi",
                "givenName": "Emerson",
                "familyName": "Fittipaldi",
                "dateOfBirth": "1946-12-12",
                "nationality": "Brazilian"
              },
              "Constructors": [
                {
                  "constructorId": "team_lotus",
                  "url": "http://en.wikipedia.org/wiki/Team_Lotus",
                  "name": "Team Lotus",
                  "nationality": "British"
                }
              ]
            },
            {
              "position": "2",
              "positionText": "2",
              "points": "45",
              "wins": "4",
              "Driver": {
                "driverId": "stewart",
                "url": "http://en.wikipedia.org/wiki/Jackie_Stewart",
                "givenName": "Jackie",
                "familyName": "Stewart",
                "dateOfBirth": "1939-06-11",
                "nationality": "British"
              },
              "Constructors": [
                {
                  "constructorId": "tyrrell",
                  "url": "http://en.wikipedia.org/wiki/Tyrrell_Racing",
                  "name": "Tyrrell",
                  "nationality": "British"
                }
              ]
            },
            ...more
          ]
        }
      ]
    }
  }
}
```

**Get Pierre Gasly's 2020 drivers standing information**

`http://api.jolpi.ca/ergast/f1/2020/drivers/gasly/driverstandings/`

`{
  "MRData": {
    "xmlns": "",
    "series": "f1",
    "url": "http://api.jolpi.ca/ergast/f1/2020/drivers/gasly/driverstandings/",
    "limit": "30",
    "offset": "0",
    "total": "1",
    "StandingsTable": {
      "season": "2020",
      "driverId": "gasly",
      "round": "17",
      "StandingsLists": [
        {
          "season": "2020",
          "round": "17",
          "DriverStandings": [
            {
              "position": "10",
              "positionText": "10",
              "points": "75",
              "wins": "1",
              "Driver": {
                "driverId": "gasly",
                "permanentNumber": "10",
                "code": "GAS",
                "url": "http://en.wikipedia.org/wiki/Pierre_Gasly",
                "givenName": "Pierre",
                "familyName": "Gasly",
                "dateOfBirth": "1996-02-07",
                "nationality": "French"
              },
              "Constructors": [
                {
                  "constructorId": "alphatauri",
                  "url": "http://en.wikipedia.org/wiki/Scuderia_AlphaTauri",
                  "name": "AlphaTauri",
                  "nationality": "Italian"
                }
              ]
            }
          ]
        }
      ]
    }
  }
}`

**Constructor Standings**

Returns a season's constructors standings from first to last place.

**URL** : `/ergast/f1/{season}/constructorstandings/`

[Available Query Parameters](https://github.com/jolpica/jolpica-f1/blob/main/docs/README.md#query-parameters)

---

**Route Parameters**

**Season (required)**

Filters for the constructors standing of a specified season. Year numbers are valid as is `current` to get the current seasons constructors standings.

`/{season}/` -> ex: `/ergast/f1/2024/constructorstandings/`

**Note**: To utilize the `season` parameter, it needs to be the first argument after `/ergast/f1/`.

---

**Round**

Filters for the constructors standings after a specified round in a specific season. Round numbers 1 -> `n` races are valid as well as `last`.

`/{season}/{round}/` -> ex: `/ergast/f1/2024/5/constructorstandings/`

**Note**: To utilize the `round` parameter it must be combined with a season filter and needs to be the first argument after `/ergast/f1/{season}/`.

---

**constructors**

Filters for only for a specific constructors' standing information for a given year.

`/constructors/{constructorsId}/` -> ex: `/ergast/f1/2024/constructors/ferrari/constructorstandings/`

---

**position**

Filters for only the constructor in a given position in a given year.

`/{finishPosition}` -> ex: `/ergast/f1/2024/constructorstandings/1/`

**Note**: The position must be at the end after any filters and after `/constructorstandings/`

---

**Success Response**

**Code** : `200 OK`

**Response Fields** :

[Common Response Fields](https://github.com/jolpica/jolpica-f1/blob/main/docs/README.md#common-response-fields)

`MRData.StandingsTable` : The object containing the season's constructors standing information.

`MRData.StandingsTable.season` : The filtered season.

`MRData.StandingsTable.round` : The round that the season the standings represent.

`MRData.StandingsTable.StandingsLists` : The list of constructors standings.

`MRData.StandingsTable.StandingsLists[i]` : A given constructors standings list object.

`MRData.StandingsTable.StandingsLists[i].ConstructorStandings` : The list of constructors standings objects.

---

**Constrcutors Standing Object Fields:**

| Field | Always Included | Description | type |
| --- | --- | --- | --- |
| position | 🟡 | Position in the Championship | String |
| positonText | ✅ | Description of position `*` | String |
| points | ✅ | Total points in the Championship | String |
| wins | ✅ | Count of race wins | String |
| Constructor | ✅ | Constructor information (constructorId, name, url, nationality) | Object |
- - Possible values for positionText include: `E` Excluded (2007 McLaren), `D` Disqualified,  for ineligible or the position as a string otherwise.

---

**Examples:**

**Get the 2007 season's constructors standing information**

`https://api.jolpi.ca/ergast/f1/2007/constructorstandings/`

```
{
  "MRData": {
    "xmlns": "",
    "series": "f1",
    "url": "http://api.jolpi.ca/ergast/f1/2007/constructorstandings/",
    "limit": "30",
    "offset": "0",
    "total": "11",
    "StandingsTable": {
      "season": "2007",
      "round": "17",
      "StandingsLists": [
        {
          "season": "2007",
          "round": "17",
          "ConstructorStandings": [
            {
              "position": "1",
              "positionText": "1",
              "points": "204",
              "wins": "9",
              "Constructor": {
                "constructorId": "ferrari",
                "url": "http://en.wikipedia.org/wiki/Scuderia_Ferrari",
                "name": "Ferrari",
                "nationality": "Italian"
              }
            },
            {
              "position": "2",
              "positionText": "2",
              "points": "101",
              "wins": "0",
              "Constructor": {
                "constructorId": "bmw_sauber",
                "url": "http://en.wikipedia.org/wiki/BMW_Sauber",
                "name": "BMW Sauber",
                "nationality": "German"
              }
            },
            ...more,
            {
              "positionText": "E",
              "points": "0",
              "wins": "8",
              "Constructor": {
                "constructorId": "mclaren",
                "url": "http://en.wikipedia.org/wiki/McLaren",
                "name": "McLaren",
                "nationality": "British"
              }
            }
          ]
        }
      ]
    }
  }
}
```

**Get the 2020 constructors standing after the 9th round**

`http://api.jolpi.ca/ergast/f1/2020/9/constructorstandings/`

`{
  "MRData": {
    "xmlns": "",
    "series": "f1",
    "url": "http://api.jolpi.ca/ergast/f1/2020/9/constructorstandings/",
    "limit": "30",
    "offset": "0",
    "total": "10",
    "StandingsTable": {
      "season": "2020",
      "round": "9",
      "StandingsLists": [
        {
          "season": "2020",
          "round": "9",
          "ConstructorStandings": [
            {
              "position": "1",
              "positionText": "1",
              "points": "325",
              "wins": "7",
              "Constructor": {
                "constructorId": "mercedes",
                "url": "http://en.wikipedia.org/wiki/Mercedes-Benz_in_Formula_One",
                "name": "Mercedes",
                "nationality": "German"
              }
            },
            {
              "position": "2",
              "positionText": "2",
              "points": "173",
              "wins": "1",
              "Constructor": {
                "constructorId": "red_bull",
                "url": "http://en.wikipedia.org/wiki/Red_Bull_Racing",
                "name": "Red Bull",
                "nationality": "Austrian"
              }
            },
            ...more
          ]
        }
      ]
    }
  }
}`

**Status**

Returns a list of finishing statuses and their counts across races matching the filters.

**Changes to Returned Data**

In the future data returned by this endpoint, and the status field returned by other endpoints may be changed.

Seasons prior to 2024 may have their corresponding status ids changed to match our updated enumeration of possible statuses ([see here](https://github.com/jolpica/jolpica-f1/blob/71f12b1c9637aa838926abcb6f4840fbfac4d87c/jolpica/formula_one/models/session.py#L64-L71)). It is recommended to rely only on the following `statusId` values: 1 (Finished), 2 (Disqualified), 3 (Accident), 31 (Retired), 143 (Lapped).

Please note statusId 3, is currently not implemented for 
newer seasons, so some statuses that currently return 31 (Retired), may 
be updated to 3 (Accident) in the future.

**URL** : `/ergast/f1/status/`

[Available Query Parameters](https://github.com/jolpica/jolpica-f1/blob/main/docs/README.md#query-parameters)

---

**Route Parameters**

**Season**

Filters only statuses from races within a specified season. Year numbers are valid as is `current`.

`/{season}/` -> ex: `/ergast/f1/2023/status/`

**Note**: To utilize the `season` parameter, it needs to be the first argument after `/ergast/f1/`.

---

**Round**

Filters only statuses from a specified round of a specific season. Round numbers 1 -> `n` races are valid as well as `last`.

`/{round}/` -> ex: `/ergast/f1/2023/10/status/`

**Note**: To utilize the `round` parameter it needs to be used with the `season` filter and be the first argument after `/ergast/f1/{season}/`.

---

**Circuits**

Filters for only statuses from races held at a given circuit.

`/circuits/{circuitId}/` -> ex: `/ergast/f1/circuits/monza/status/`

---

**Constructors**

Filters for only statuses achieved by drivers driving for a specified constructor.

`/constructors/{constructorId}/` -> ex: `/ergast/f1/constructors/ferrari/status/`

---

**Drivers**

Filters for only statuses achieved by a specified driver.

`/drivers/{driverId}/` -> ex: `/ergast/f1/drivers/alonso/status/`

---

**Grid**

Filters for only statuses from drivers who started a race in a specific grid position.

`/grid/{gridPosition}/` -> ex: `/ergast/f1/grid/1/status/`

---

**Results**

Filters for only statuses from drivers who finished a race in a specific position.

`/results/{finishPosition}/` -> ex: `/ergast/f1/results/1/status/`

---

**Fastest**

Filters for only statuses from drivers who achieved a specific fastest lap rank in a race.

`/fastest/{lapRank}/` -> ex: `/ergast/f1/fastest/1/status/`

---

**status**

Filters for only a specific status ID.

`/status/{statusId}/` -> ex: `/ergast/f1/status/1/`

**DISCLAIMER**: Seasons prior to 2024 may have their corresponding status ids changed to match our updated enumeration of possible statuses ([see here](https://github.com/jolpica/jolpica-f1/blob/71f12b1c9637aa838926abcb6f4840fbfac4d87c/jolpica/formula_one/models/session.py#L64-L71)). It is recommended to rely only on the following `statusId` values: 1 (Finished), 2 (Disqualified), 3 (Accident), 31 (Retired), 143 (Lapped).

Please note `statusId` 3, is currently not 
implemented for newer seasons, so some statuses that currently return 31
 (Retired), may be updated to 3 (Accident) in the future.

---

**Success Response**

**Code** : `200 OK`

**Response Fields** :

[Common Response Fields](https://github.com/jolpica/jolpica-f1/blob/main/docs/README.md#common-response-fields)

`MRData.StatusTable` : The object containing the list of statuses.

`MRData.StatusTable.Status` : The list of all statuses returned, ordered by count descending.

`MRData.StatusTable.Status[i]` : A given status object.

---

**Status Object Fields:**

| Field | Always Included | Description | type |
| --- | --- | --- | --- |
| statusId | ✅ | Unique ID of the status | String |
| count | ✅ | Number of times this status occurred within the filtered races | String |
| status | ✅ | Description of the status | String |

---

**Examples:**

**Get list of all statuses in F1 history**

`https://api.jolpi.ca/ergast/f1/status/`

```
{
  "MRData": {
    "xmlns": "",
    "series": "f1",
    "url": "http://api.jolpi.ca/ergast/f1/status/",
    "limit": "30",
    "offset": "0",
    "total": "138",
    "StatusTable": {
      "Status": [
        {
          "statusId": "1",
          "count": "14785",
          "status": "Finished"
        },
        {
          "statusId": "11",
          "count": "2986",
          "status": "+1 Lap"
        },
        {
          "statusId": "12",
          "count": "1409",
          "status": "+2 Laps"
        },
        {
            "statusId": "5",
            "count": "977",
            "status": "Engine"
        },
        ...more
      ]
    }
  }
}
```

**Get all statuses achieved by Ferrari in the 2024 season**

`https://api.jolpi.ca/ergast/f1/2024/constructors/ferrari/status/`

```
{
    "MRData": {
        "xmlns": "",
        "series": "f1",
        "url": "http://api.jolpi.ca/ergast/f1/2024/constructors/ferrari/status/",
        "limit": "30",
        "offset": "0",
        "total": "3",
        "StatusTable": {
            "season": "2024",
            "constructorId": "ferrari",
            "Status": [
                {
                    "statusId": "1",
                    "count": "44",
                    "status": "Finished"
                },
                {
                    "statusId": "31",
                    "count": "3",
                    "status": "Retired"
                },
                {
                    "statusId": "143",
                    "count": "1",
                    "status": "Lapped"
                }
            ]
        }
    }
}
```