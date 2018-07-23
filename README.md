# Linto-command-module
Linto Command Module is the module in charge of:
* Detecting keyword
* Detecting beginning and ending of voice command utterances.

Disclaimer: This repository is an early stage development and may heavily vary or be replaced within short notice.

## Getting Started

To get a copy a this repository:
```
git clone ...
```

The project 
## Dependencies

This program requires python3 and pip to work.
```
sudo apt-get install python3 python3-pip
```
Python libraries required to run the module can be found within the requirements.txt file and can be installed at once using:
```
sudo pip3 install -r requirements.txt
```

### HOW TO
## Architecture
<img src="https://image.ibb.co/eHBAPJ/voice_commander_diagram.png"
     alt="project architecture"
     style="float: center;" />
     

## KeyWord and models.
The model provided with the repository is a model to spot the "Linto" Keyword based on ~500 recording made by a dozen of persons.
If you want to use an other KeyWord please refer to [mycroft-precise wiki](https://github.com/MycroftAI/mycroft-precise/wiki/Training-your-own-wake-word).


## Built With

* [Mycroft-Precise](https://github.com/MycroftAI/mycroft-precise) - A lightweight, simple-to-use, RNN wake word listener.
* [Mosquitto](https://mosquitto.org/) - Easy to use MQTT Broker
* [Tenacity](https://github.com/jd/tenacity) - General-purpose retrying library
* [paho-mqtt](https://pypi.org/project/paho-mqtt/) - MQTT client library.


## License

This project is licensed under the GNU AFFERO License - see the [LICENSE.md](LICENSE.md) file for details

