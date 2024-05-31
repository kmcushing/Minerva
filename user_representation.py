# could initially just be a json with most common topics (and some measure of their frequency) along with a list of domains they have insider info on
import jsons


class User:
    def __init__(self, id, insider_domains=None, topic_frequencies=None):
        self.id = id
        self.insider_domains = insider_domains if insider_domains else {}
        self.topic_frequencies = topic_frequencies if topic_frequencies else {}
        self.save()

    def discussed_topic(self, topic):
        # adds to discussion count of topic in user representation
        if topic in self.topic_frequencies:
            self.topic_frequencies[topic] += 1
        else:
            self.topic_frequencies[topic] = 1
        self.save()

    def add_insider_domain(self, domain):
        # adds insider domain for user
        self.insider_domains[domain] = 1
        self.save()

    def remove_insider_domain(self, domain):
        self.insider_domains.pop(domain, None)
        self.save()

    def __str__(self):
        return f"Insider Domains: {self.insider_domains}\nTopic Frequencies: {self.topic_frequencies}"

    def to_prompt(self):
        # returns text representation of user for llm prompt
        s = f"The user's name is {self.id}. Assume they do not have insider knowledge for all domains unless otherwise specified. "
        if self.topic_frequencies:
            s += "The user has previously discuessed the following topics:\n"
            for t in self.topic_frequencies:
                s += f"{t}: Frequency {self.topic_frequencies[t]}\n"
        if self.insider_domains:
            s += "The user has insider knowledge of the following domains:\n"
            for d in self.insider_domains:
                s += f"{d}\n"
        return s

    def save(self):
        f = open(f"data/user_representations/{self.id}.json", "w")
        f.write(jsons.dumps(self))


# load user representation
def load_user(id):
    try:
        f = open(f"data/user_representations/{id}.json", "r")
        return jsons.loads(f.read(), User)
    except:
        return User(id)


# test loading user
# user1 = load_user("user1")
# print(user1.to_prompt())
