import unittest


class MyTestCase(unittest.TestCase):
    def test_attrdict(self):
        from pytools.pyutils.misc.nested import nested_dictify

        source = {
            "apple": {
                "weight": 300,
                "attributes": [
                    {
                        "seller": "Metro"
                    },
                    {
                        "seller": "Amazon"
                    }
                ]
            }
        }

        target = nested_dictify(source)
        target.set_fallback("none")
        self.assertEqual(target.apple.weight, source['apple']['weight'])
        self.assertEqual(target.apple.attributes[0].seller, source['apple']['attributes'][0]['seller'])
        self.assertEqual(target.notexist, "none")
        self.assertEqual(getattr(target, "notexist"), "none")


if __name__ == '__main__':
    unittest.main()
